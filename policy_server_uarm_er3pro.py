import argparse
import math
import queue
import threading
import time
from collections import deque

import cv2 as cv
import dill
import hydra
import numpy as np
import torch
import zmq

from diffusion_policy.common.pytorch_util import dict_apply


POLICY_CONTROL_PERIOD = 0.10
POLICY_ACTION_PERIOD = 0.10
LATENCY_BUDGET = 0.30
LATENCY_STEPS = math.ceil(LATENCY_BUDGET / POLICY_CONTROL_PERIOD)
ACTION_REPEAT = max(1, int(round(POLICY_ACTION_PERIOD / POLICY_CONTROL_PERIOD)))
PROFILE_INTERVAL = 2.0
UARM_JOINT_LIMIT_RAD_MIN = np.deg2rad(
    np.array([-170.0, -120.0, -170.0, -170.0, -170.0, -170.0, -170.0], dtype=np.float64)
)
UARM_JOINT_LIMIT_RAD_MAX = np.deg2rad(
    np.array([170.0, 120.0, 170.0, 170.0, 170.0, 170.0, 170.0], dtype=np.float64)
)
UARM_MAX_JOINT_STEP_RAD = np.deg2rad(
    np.array([6.0, 6.0, 6.0, 9.0, 9.0, 9.0, 9.0], dtype=np.float64)
)
UARM_MAX_GRIPPER_STEP = 0.08


class UArmEr3ProDiffusionPolicy:
    def __init__(self, ckpt_path, device="cuda"):
        with open(ckpt_path, "rb") as f:
            payload = torch.load(f, pickle_module=dill, map_location=device)

        cfg = payload["cfg"]
        workspace_cls = hydra.utils.get_class(cfg._target_)
        workspace = workspace_cls(cfg)
        workspace.load_payload(payload)

        policy = workspace.ema_model if cfg.training.use_ema else workspace.model
        if device.startswith("cuda"):
            torch.backends.cudnn.benchmark = True
        self.device = torch.device(device)
        self.policy = policy.eval().to(self.device)
        self.obs_shape_meta = cfg.shape_meta["obs"]
        self.n_obs_steps = int(cfg.policy.n_obs_steps)
        self.n_action_steps = int(cfg.policy.n_action_steps)
        self.warmed_up = False
        self.profile_last_time = time.time()
        self.profile_count = 0
        self.profile_total_ms = 0.0
        self.profile_max_ms = 0.0
        print("[uarm_policy] checkpoint observation shapes:", flush=True)
        for key, value in self.obs_shape_meta.items():
            print(
                f"  {key}: type={value.get('type', 'low_dim')} shape={list(value['shape'])}",
                flush=True,
            )
        print(
            f"[uarm_policy] n_obs_steps={self.n_obs_steps} "
            f"n_action_steps={self.n_action_steps}",
            flush=True,
        )

    def reset(self):
        self.policy.reset()

    def step(self, obs_sequence):
        obs_dict = self._convert_obs(obs_sequence)
        with torch.inference_mode():
            if not self.warmed_up:
                print("[uarm_policy] warming up", flush=True)
                self.policy.predict_action(obs_dict)
                self.warmed_up = True

            t0 = time.time()
            result = self.policy.predict_action(obs_dict)
            infer_ms = 1000.0 * (time.time() - t0)
            self._record_profile(infer_ms)

        action = result["action"][0].detach().cpu().numpy()
        return self._convert_action(action, obs_sequence[-1])

    def _convert_obs(self, obs_sequence):
        obs_dict_np = {}
        for key, value in self.obs_shape_meta.items():
            if key not in obs_sequence[-1] and key != "robot_state":
                raise KeyError(f"missing observation key required by checkpoint: {key}")

            if value.get("type") == "rgb":
                target_shape = tuple(value["shape"])
                images = np.stack(
                    [self._prepare_rgb_obs(obs[key], target_shape, key) for obs in obs_sequence],
                    axis=0,
                )
                images = images.astype(np.float32) / 255.0
                images = np.transpose(images, (0, 3, 1, 2))
                if images.shape[1:] != target_shape:
                    raise ValueError(f"{key} shape {images.shape[1:]} != {target_shape}")
                obs_dict_np[key] = images
            elif key == "robot_state":
                robot_state = np.stack([self._robot_state_from_obs(obs) for obs in obs_sequence], axis=0)
                expected_shape = tuple(value["shape"])
                if robot_state.shape[1:] != expected_shape:
                    raise ValueError(f"{key} shape {robot_state.shape[1:]} != {expected_shape}")
                obs_dict_np[key] = robot_state
            else:
                low_dim = np.stack([obs[key] for obs in obs_sequence], axis=0).astype(np.float32)
                expected_shape = tuple(value["shape"])
                if low_dim.shape[1:] != expected_shape:
                    raise ValueError(f"{key} shape {low_dim.shape[1:]} != {expected_shape}")
                obs_dict_np[key] = low_dim

        return dict_apply(obs_dict_np, lambda x: torch.from_numpy(x).unsqueeze(0).to(self.device))

    @staticmethod
    def _prepare_rgb_obs(image, target_chw_shape, key):
        if len(target_chw_shape) != 3 or target_chw_shape[0] != 3:
            raise ValueError(f"{key} expected RGB CHW shape [3, H, W], got {target_chw_shape}")

        image = np.asarray(image)
        target_h, target_w = target_chw_shape[1], target_chw_shape[2]

        if image.ndim == 3 and image.shape[0] == 3 and image.shape[2] != 3:
            image = np.transpose(image, (1, 2, 0))

        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"{key} expected HWC RGB image, got shape {image.shape}")

        if image.dtype != np.uint8:
            if np.issubdtype(image.dtype, np.floating) and image.max(initial=0.0) <= 1.0:
                image = np.clip(image * 255.0, 0, 255).astype(np.uint8)
            else:
                image = np.clip(image, 0, 255).astype(np.uint8)

        if image.shape[:2] != (target_h, target_w):
            image = cv.resize(image, (target_w, target_h), interpolation=cv.INTER_AREA)

        return np.ascontiguousarray(image)

    @staticmethod
    def _robot_state_from_obs(obs):
        if "robot_state" in obs:
            return np.asarray(obs["robot_state"], dtype=np.float32)

        return np.concatenate(
            [
                np.asarray(obs["arm_joints"], dtype=np.float32).reshape(7),
                np.asarray(obs["arm_pos"], dtype=np.float32).reshape(3),
                np.asarray(obs["arm_quat"], dtype=np.float32).reshape(4),
                np.asarray(obs["gripper_pos"], dtype=np.float32).reshape(1),
            ],
            axis=0,
        )

    @staticmethod
    def _convert_action(action, latest_obs):
        act_sequence = []
        current_joints, current_gripper = UArmEr3ProDiffusionPolicy._current_command_from_obs(latest_obs)
        for act in action:
            act = np.asarray(act, dtype=np.float32)
            if act.shape[0] < 8:
                raise ValueError(f"policy action shape {act.shape} does not contain 7 joints + gripper")

            raw_joints = act[:7].astype(np.float64)
            raw_gripper = act[7:8].astype(np.float64)

            if np.all(np.isfinite(raw_joints)):
                target_joints = np.clip(raw_joints, UARM_JOINT_LIMIT_RAD_MIN, UARM_JOINT_LIMIT_RAD_MAX)
                joint_delta = np.clip(
                    target_joints - current_joints,
                    -UARM_MAX_JOINT_STEP_RAD,
                    UARM_MAX_JOINT_STEP_RAD,
                )
                current_joints = np.clip(
                    current_joints + joint_delta,
                    UARM_JOINT_LIMIT_RAD_MIN,
                    UARM_JOINT_LIMIT_RAD_MAX,
                )

            if np.all(np.isfinite(raw_gripper)):
                target_gripper = float(np.clip(raw_gripper[0], 0.0, 1.0))
                gripper_delta = np.clip(
                    target_gripper - current_gripper,
                    -UARM_MAX_GRIPPER_STEP,
                    UARM_MAX_GRIPPER_STEP,
                )
                current_gripper = float(np.clip(current_gripper + gripper_delta, 0.0, 1.0))

            act_sequence.append(
                {
                    "arm_joints": current_joints.astype(np.float64).copy(),
                    "gripper_pos": np.asarray([current_gripper], dtype=np.float64),
                }
            )
        return act_sequence

    @staticmethod
    def _current_command_from_obs(obs):
        if "arm_joints" in obs:
            joints = np.asarray(obs["arm_joints"], dtype=np.float64).reshape(7)
        elif "robot_state" in obs:
            joints = np.asarray(obs["robot_state"], dtype=np.float64).reshape(-1)[:7]
        else:
            raise KeyError("missing arm_joints/robot_state needed to safety-filter policy action")

        if "gripper_pos" in obs:
            gripper = float(np.asarray(obs["gripper_pos"], dtype=np.float64).reshape(-1)[0])
        elif "robot_state" in obs:
            gripper = float(np.asarray(obs["robot_state"], dtype=np.float64).reshape(-1)[14])
        else:
            gripper = 0.0

        if not np.all(np.isfinite(joints)):
            raise ValueError(f"latest arm_joints contains non-finite values: {joints}")
        if not np.isfinite(gripper):
            gripper = 0.0

        joints = np.clip(joints, UARM_JOINT_LIMIT_RAD_MIN, UARM_JOINT_LIMIT_RAD_MAX)
        gripper = float(np.clip(gripper, 0.0, 1.0))
        return joints, gripper

    def _record_profile(self, infer_ms):
        self.profile_count += 1
        self.profile_total_ms += infer_ms
        self.profile_max_ms = max(self.profile_max_ms, infer_ms)
        now = time.time()
        dt = now - self.profile_last_time
        if dt < PROFILE_INTERVAL:
            return
        print(
            f"[uarm_policy] infer_hz={self.profile_count / dt:.1f} "
            f"avg_infer_ms={self.profile_total_ms / self.profile_count:.1f} "
            f"max_infer_ms={self.profile_max_ms:.1f}",
            flush=True,
        )
        self.profile_last_time = now
        self.profile_count = 0
        self.profile_total_ms = 0.0
        self.profile_max_ms = 0.0


class PolicyWrapper:
    def __init__(
        self,
        policy,
        n_obs_steps=None,
        n_action_steps=None,
        latency_steps=LATENCY_STEPS,
        action_repeat=ACTION_REPEAT,
        control_period=POLICY_CONTROL_PERIOD,
        latency_budget=LATENCY_BUDGET,
    ):
        self.n_obs_steps = int(n_obs_steps if n_obs_steps is not None else policy.n_obs_steps)
        self.n_action_steps = int(n_action_steps if n_action_steps is not None else policy.n_action_steps)
        self.latency_steps = int(max(0, latency_steps))
        if self.latency_steps >= self.n_action_steps:
            print(
                f"[uarm_policy] latency_steps={self.latency_steps} >= n_action_steps={self.n_action_steps}; "
                f"clamping to {self.n_action_steps - 1}",
                flush=True,
            )
            self.latency_steps = self.n_action_steps - 1
        self.action_repeat = int(max(1, action_repeat))
        self.control_period = float(control_period)
        self.latency_budget = float(latency_budget)
        self.obs_queue = queue.Queue()
        self.act_queue = queue.Queue()
        self.last_error = None
        self.profile_last_time = time.time()
        self.profile_infer_count = 0
        self.profile_empty_count = 0
        self.profile_drop_count = 0
        print(
            f"[uarm_policy] wrapper n_obs_steps={self.n_obs_steps} "
            f"n_action_steps={self.n_action_steps} latency_steps={self.latency_steps} "
            f"action_repeat={self.action_repeat} control_period={self.control_period:.3f}s "
            f"latency_budget={self.latency_budget:.3f}s",
            flush=True,
        )
        threading.Thread(target=self.inference_loop, args=(policy,), daemon=True).start()

    def reset(self):
        self.last_error = None
        self.obs_queue.put("reset")

    def step(self, obs):
        if self.last_error is not None:
            error = self.last_error
            self.last_error = None
            raise RuntimeError(f"policy inference loop failed: {error}")

        self.obs_queue.put(obs)
        if self.act_queue.empty():
            self.profile_empty_count += 1
            return None
        return self.act_queue.get()

    def inference_loop(self, policy):
        obs_history = deque(maxlen=self.n_obs_steps)
        start_of_episode = True
        while True:
            latest_obs = None
            reset_requested = False
            dropped_obs = 0
            while not self.obs_queue.empty():
                obs = self.obs_queue.get()
                if obs == "reset":
                    reset_requested = True
                    latest_obs = None
                else:
                    if latest_obs is not None:
                        dropped_obs += 1
                    latest_obs = obs

            if dropped_obs:
                self.profile_drop_count += dropped_obs

            if reset_requested:
                policy.reset()
                obs_history.clear()
                start_of_episode = True
                self.last_error = None
                while not self.act_queue.empty():
                    self.act_queue.get()

            if latest_obs is not None:
                obs_history.append(latest_obs)

            if self.act_queue.qsize() <= self.latency_steps and len(obs_history) == self.n_obs_steps:
                try:
                    self.profile_infer_count += 1
                    act_sequence = policy.step(list(obs_history))
                except Exception as e:
                    self.last_error = e
                    print(f"[uarm_policy] inference error: {type(e).__name__}: {e}", flush=True)
                    obs_history.clear()
                    while not self.act_queue.empty():
                        self.act_queue.get()
                    time.sleep(0.1)
                    continue
                if start_of_episode:
                    act_sequence = act_sequence[: max(1, self.n_action_steps - self.latency_steps)]
                    start_of_episode = False
                else:
                    act_sequence = act_sequence[self.latency_steps : self.n_action_steps]
                for action in act_sequence:
                    for _ in range(self.action_repeat):
                        self.act_queue.put(action)

            self._maybe_print_profile()
            time.sleep(0.001)

    def _maybe_print_profile(self):
        now = time.time()
        dt = now - self.profile_last_time
        if dt < PROFILE_INTERVAL:
            return
        if (
            self.profile_infer_count == 0
            and self.profile_empty_count == 0
            and self.profile_drop_count == 0
            and self.act_queue.empty()
        ):
            self.profile_last_time = now
            return
        print(
            f"[uarm_queue] infer_hz={self.profile_infer_count / dt:.1f} "
            f"queue={self.act_queue.qsize()} empty={self.profile_empty_count} "
            f"dropped_obs={self.profile_drop_count}",
            flush=True,
        )
        self.profile_last_time = now
        self.profile_infer_count = 0
        self.profile_empty_count = 0
        self.profile_drop_count = 0


class PolicyServer:
    def __init__(self, policy, port=5555):
        self.policy = policy
        context = zmq.Context()
        self.socket = context.socket(zmq.REP)
        self.socket.bind(f"tcp://*:{port}")
        print(f"[uarm_policy] server started on port {port}", flush=True)

    def run(self):
        while True:
            req = self.socket.recv_pyobj()
            rep = {}
            try:
                if "reset" in req:
                    self.policy.reset()
                    print("[uarm_policy] reset", flush=True)
                elif "obs" in req:
                    rep["action"] = self.step(req["obs"])
            except Exception as e:
                rep["error"] = str(e)
                print(f"[uarm_policy] error: {e}", flush=True)
            self.socket.send_pyobj(rep)

    def step(self, obs):
        for key, value in list(obs.items()):
            if key.endswith("image"):
                if isinstance(value, np.ndarray) and value.dtype == np.uint8 and (
                    value.ndim == 1 or (value.ndim == 2 and 1 in value.shape)
                ):
                    bgr = cv.imdecode(value, cv.IMREAD_COLOR)
                    if bgr is None:
                        raise RuntimeError(f"failed to decode image key {key}")
                    obs[key] = cv.cvtColor(bgr, cv.COLOR_BGR2RGB)
                else:
                    obs[key] = value
        return self.policy.step(obs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-path", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument(
        "--control-period",
        type=float,
        default=POLICY_CONTROL_PERIOD,
        help="Robot policy-control period in seconds. Must match tidybot2 POLICY_CONTROL_PERIOD.",
    )
    parser.add_argument(
        "--latency-budget",
        type=float,
        default=LATENCY_BUDGET,
        help="Latency to hide in seconds. Use ceil(latency_budget / control_period) action steps.",
    )
    args = parser.parse_args()

    if args.control_period <= 0:
        raise ValueError("--control-period must be positive")
    if args.latency_budget < 0:
        raise ValueError("--latency-budget must be non-negative")

    latency_steps = math.ceil(args.latency_budget / args.control_period)
    action_repeat = max(1, int(round(POLICY_ACTION_PERIOD / args.control_period)))
    policy = PolicyWrapper(
        UArmEr3ProDiffusionPolicy(args.ckpt_path, device=args.device),
        latency_steps=latency_steps,
        action_repeat=action_repeat,
        control_period=args.control_period,
        latency_budget=args.latency_budget,
    )
    PolicyServer(policy, port=args.port).run()


if __name__ == "__main__":
    main()
