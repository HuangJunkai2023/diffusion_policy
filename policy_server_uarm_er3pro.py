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


POLICY_CONTROL_PERIOD = 0.05
LATENCY_BUDGET = 0.20
LATENCY_STEPS = math.ceil(LATENCY_BUDGET / POLICY_CONTROL_PERIOD)
PROFILE_INTERVAL = 2.0


class UArmEr3ProDiffusionPolicy:
    def __init__(self, ckpt_path, device="cuda"):
        with open(ckpt_path, "rb") as f:
            payload = torch.load(f, pickle_module=dill, map_location=device)

        cfg = payload["cfg"]
        workspace_cls = hydra.utils.get_class(cfg._target_)
        workspace = workspace_cls(cfg)
        workspace.load_payload(payload)

        policy = workspace.ema_model if cfg.training.use_ema else workspace.model
        self.device = torch.device(device)
        self.policy = policy.eval().to(self.device)
        self.obs_shape_meta = cfg.shape_meta["obs"]
        self.warmed_up = False
        self.profile_last_time = time.time()
        self.profile_count = 0
        self.profile_total_ms = 0.0
        self.profile_max_ms = 0.0

    def reset(self):
        self.policy.reset()

    def step(self, obs_sequence):
        obs_dict = self._convert_obs(obs_sequence)
        with torch.no_grad():
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
            if value.get("type") == "rgb":
                images = np.stack([obs[key] for obs in obs_sequence], axis=0)
                if images.dtype != np.uint8:
                    raise TypeError(f"{key} must be uint8, got {images.dtype}")
                images = images.astype(np.float32) / 255.0
                images = np.transpose(images, (0, 3, 1, 2))
                if images.shape[1:] != tuple(value["shape"]):
                    raise ValueError(f"{key} shape {images.shape[1:]} != {tuple(value['shape'])}")
                obs_dict_np[key] = images
            elif key == "robot_state":
                obs_dict_np[key] = np.stack([self._robot_state_from_obs(obs) for obs in obs_sequence], axis=0)
            else:
                obs_dict_np[key] = np.stack([obs[key] for obs in obs_sequence], axis=0).astype(np.float32)

        return dict_apply(obs_dict_np, lambda x: torch.from_numpy(x).unsqueeze(0).to(self.device))

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
        for act in action:
            act = np.asarray(act, dtype=np.float32)
            act_sequence.append(
                {
                    "arm_joints": act[:7].astype(np.float64),
                    "arm_pos": np.asarray(latest_obs["arm_pos"], dtype=np.float64).copy(),
                    "arm_quat": np.asarray(latest_obs["arm_quat"], dtype=np.float64).copy(),
                    "gripper_pos": np.asarray(act[7:8], dtype=np.float64),
                }
            )
        return act_sequence

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
    def __init__(self, policy, n_obs_steps=2, n_action_steps=8):
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.obs_queue = queue.Queue()
        self.act_queue = queue.Queue()
        threading.Thread(target=self.inference_loop, args=(policy,), daemon=True).start()

    def reset(self):
        self.obs_queue.put("reset")

    def step(self, obs):
        self.obs_queue.put(obs)
        if self.act_queue.empty():
            print("[uarm_policy] action queue empty; holding this cycle", flush=True)
            return None
        return self.act_queue.get()

    def inference_loop(self, policy):
        obs_history = deque(maxlen=self.n_obs_steps)
        start_of_episode = True
        while True:
            if not self.obs_queue.empty():
                obs = self.obs_queue.get()
                if obs == "reset":
                    policy.reset()
                    obs_history.clear()
                    start_of_episode = True
                    while not self.act_queue.empty():
                        self.act_queue.get()
                    continue
                obs_history.append(obs)

            if self.act_queue.qsize() < LATENCY_STEPS and len(obs_history) == self.n_obs_steps:
                act_sequence = policy.step(list(obs_history))
                if start_of_episode:
                    act_sequence = act_sequence[: self.n_action_steps - LATENCY_STEPS]
                    start_of_episode = False
                else:
                    act_sequence = act_sequence[LATENCY_STEPS : self.n_action_steps]
                for action in act_sequence:
                    self.act_queue.put(action)

            time.sleep(0.001)


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
                bgr = cv.imdecode(value, cv.IMREAD_COLOR)
                if bgr is None:
                    raise RuntimeError(f"failed to decode image key {key}")
                obs[key] = cv.cvtColor(bgr, cv.COLOR_BGR2RGB)
        return self.policy.step(obs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-path", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--port", type=int, default=5555)
    args = parser.parse_args()

    policy = PolicyWrapper(UArmEr3ProDiffusionPolicy(args.ckpt_path, device=args.device))
    PolicyServer(policy, port=args.port).run()


if __name__ == "__main__":
    main()
