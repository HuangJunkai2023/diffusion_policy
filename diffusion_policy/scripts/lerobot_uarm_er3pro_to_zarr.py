if __name__ == "__main__":
    import pathlib
    import sys

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)

import argparse
import json
import shutil
from pathlib import Path

import cv2
import numcodecs
import numpy as np
import pyarrow.parquet as pq
import zarr


DATA_COLUMNS = ["observation.state", "action", "timestamp", "frame_index", "episode_index", "index"]
BASE_VIDEO_KEY = "observation.images.base"
WRIST_VIDEO_KEY = "observation.images.wrist"


def is_lerobot_root(path: Path) -> bool:
    return (path / "meta" / "info.json").is_file() and (path / "data").is_dir()


def find_lerobot_roots(path: Path) -> list[Path]:
    if is_lerobot_root(path):
        return [path]
    roots = [p for p in sorted(path.iterdir()) if p.is_dir() and is_lerobot_root(p)]
    return roots


def read_info(root: Path) -> dict:
    return json.loads((root / "meta" / "info.json").read_text())


def episode_files(root: Path) -> list[Path]:
    files = sorted((root / "meta" / "episodes").glob("chunk-*/*.parquet"))
    return files


def load_episodes(root: Path):
    rows = []
    for path in episode_files(root):
        table = pq.read_table(path)
        rows.extend(table.to_pylist())
    rows.sort(key=lambda row: int(row["episode_index"]))
    return rows


def parquet_path(root: Path, kind: str, chunk_index: int, file_index: int) -> Path:
    return root / kind / f"chunk-{chunk_index:03d}" / f"file-{file_index:03d}.parquet"


def video_path(root: Path, video_key: str, chunk_index: int, file_index: int) -> Path:
    return root / "videos" / video_key / f"chunk-{chunk_index:03d}" / f"file-{file_index:03d}.mp4"


def read_data_rows(root: Path, episode: dict):
    path = parquet_path(root, "data", int(episode["data/chunk_index"]), int(episode["data/file_index"]))
    table = pq.read_table(path, columns=DATA_COLUMNS)
    data = table.to_pydict()
    episode_index = int(episode["episode_index"])
    mask = np.asarray(data["episode_index"], dtype=np.int64) == episode_index
    state = np.asarray(data["observation.state"], dtype=np.float32)[mask]
    action = np.asarray(data["action"], dtype=np.float32)[mask]
    return state, action


def read_video_frames(root: Path, episode: dict, video_key: str, size: tuple[int, int], fps: float):
    chunk_idx = int(episode[f"videos/{video_key}/chunk_index"])
    file_idx = int(episode[f"videos/{video_key}/file_index"])
    from_ts = float(episode[f"videos/{video_key}/from_timestamp"])
    length = int(episode["length"])
    start_frame = int(round(from_ts * fps))
    path = video_path(root, video_key, chunk_idx, file_idx)

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"failed to open video: {path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frames = []
    width, height = size
    for _ in range(length):
        ok, frame = cap.read()
        if not ok:
            cap.release()
            raise RuntimeError(f"failed to read frame {len(frames)} from {path}")
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if (frame.shape[1], frame.shape[0]) != (width, height):
            frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
        frames.append(frame)
    cap.release()
    return np.asarray(frames, dtype=np.uint8)


def collect_episodes(
    roots: list[Path],
    image_size: tuple[int, int],
    min_episode_frames: int = 2,
    skip_bad_episodes: bool = True,
):
    states = []
    actions = []
    base_images = []
    wrist_images = []
    episode_ends = []
    total_frames = 0

    for root in roots:
        info = read_info(root)
        fps = float(info["fps"])
        episodes = load_episodes(root)
        if not episodes:
            print(f"SKIP {root}: no episode metadata")
            continue

        print(f"READ {root}: {len(episodes)} episodes, {info.get('total_frames')} frames")
        for episode in episodes:
            length = int(episode["length"])
            episode_name = f"{root.name}/episode-{int(episode['episode_index']):06d}"
            if length < min_episode_frames:
                print(f"SKIP {episode_name}: length {length} < {min_episode_frames}")
                continue

            try:
                state, action = read_data_rows(root, episode)
                if len(state) != length or len(action) != length:
                    raise RuntimeError(
                        f"length mismatch: meta={length} state={len(state)} action={len(action)}"
                    )
                base = read_video_frames(root, episode, BASE_VIDEO_KEY, image_size, fps)
                wrist = read_video_frames(root, episode, WRIST_VIDEO_KEY, image_size, fps)
            except Exception as exc:
                if skip_bad_episodes:
                    print(f"SKIP {episode_name}: {exc}")
                    continue
                raise

            states.append(state)
            actions.append(action)
            base_images.append(base)
            wrist_images.append(wrist)
            total_frames += length
            episode_ends.append(total_frames)

    if total_frames == 0:
        raise RuntimeError("no complete episodes found")

    return {
        "robot_state": np.concatenate(states, axis=0).astype(np.float32),
        "action": np.concatenate(actions, axis=0).astype(np.float32),
        "base_image": np.concatenate(base_images, axis=0).astype(np.uint8),
        "wrist_image": np.concatenate(wrist_images, axis=0).astype(np.uint8),
        "episode_ends": np.asarray(episode_ends, dtype=np.int64),
    }


def write_zarr(output: Path, arrays: dict):
    if output.exists():
        shutil.rmtree(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    root = zarr.open(str(output), mode="w")
    data_group = root.create_group("data")
    meta_group = root.create_group("meta")
    compressor = numcodecs.Blosc(cname="zstd", clevel=3, shuffle=numcodecs.Blosc.BITSHUFFLE)
    n = int(arrays["episode_ends"][-1])

    for key in ["robot_state", "action", "base_image", "wrist_image"]:
        value = arrays[key]
        if value.shape[0] != n:
            raise RuntimeError(f"{key} first dimension {value.shape[0]} != total frames {n}")
        chunks = (min(64, n),) + value.shape[1:] if value.ndim == 4 else (min(1024, n), value.shape[1])
        data_group.array(key, value, chunks=chunks, compressor=compressor, overwrite=True)
        print(f"WRITE data/{key}: shape={value.shape} dtype={value.dtype}")

    meta_group.array("episode_ends", arrays["episode_ends"], chunks=(len(arrays["episode_ends"]),), overwrite=True)
    print(f"WRITE meta/episode_ends: {arrays['episode_ends'].tolist()}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", default="data/lerobot_uarm_er3pro")
    parser.add_argument("--output", "-o", default="src/diffusion_policy/data/uarm_er3pro/uarm_er3pro_replay.zarr")
    parser.add_argument("--image-width", type=int, default=84)
    parser.add_argument("--image-height", type=int, default=84)
    parser.add_argument("--min-episode-frames", type=int, default=2)
    parser.add_argument("--skip-bad-episodes", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    roots = find_lerobot_roots(Path(args.input).expanduser())
    if not roots:
        raise RuntimeError(f"no LeRobot roots found under {args.input}")
    arrays = collect_episodes(
        roots,
        image_size=(args.image_width, args.image_height),
        min_episode_frames=args.min_episode_frames,
        skip_bad_episodes=args.skip_bad_episodes,
    )
    write_zarr(Path(args.output).expanduser(), arrays)


if __name__ == "__main__":
    main()
