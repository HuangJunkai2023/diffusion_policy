from typing import Dict
import copy

import numpy as np
import torch

from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import SequenceSampler, downsample_mask, get_val_mask
from diffusion_policy.common.normalize_util import get_image_range_normalizer
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer


class UArmEr3ProImageDataset(BaseImageDataset):
    def __init__(
        self,
        zarr_path,
        horizon=16,
        pad_before=0,
        pad_after=0,
        n_obs_steps=None,
        seed=42,
        val_ratio=0.0,
        max_train_episodes=None,
    ):
        super().__init__()
        self.replay_buffer = ReplayBuffer.copy_from_path(
            zarr_path,
            keys=["base_image", "wrist_image", "robot_state", "action"],
        )

        val_mask = get_val_mask(
            n_episodes=self.replay_buffer.n_episodes,
            val_ratio=val_ratio,
            seed=seed,
        )
        train_mask = downsample_mask(
            mask=~val_mask,
            max_n=max_train_episodes,
            seed=seed,
        )

        key_first_k = None
        if n_obs_steps is not None:
            key_first_k = {
                "base_image": n_obs_steps,
                "wrist_image": n_obs_steps,
                "robot_state": n_obs_steps,
            }

        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=horizon,
            pad_before=pad_before,
            pad_after=pad_after,
            episode_mask=train_mask,
            key_first_k=key_first_k,
        )
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.n_obs_steps = n_obs_steps

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self.horizon,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            episode_mask=~self.train_mask,
        )
        val_set.train_mask = ~self.train_mask
        return val_set

    def get_normalizer(self, **kwargs) -> LinearNormalizer:
        normalizer = LinearNormalizer()
        normalizer["action"] = SingleFieldLinearNormalizer.create_fit(self.replay_buffer["action"])
        normalizer["robot_state"] = SingleFieldLinearNormalizer.create_fit(self.replay_buffer["robot_state"])
        normalizer["base_image"] = get_image_range_normalizer()
        normalizer["wrist_image"] = get_image_range_normalizer()
        return normalizer

    def get_all_actions(self) -> torch.Tensor:
        return torch.from_numpy(self.replay_buffer["action"])

    def __len__(self) -> int:
        return len(self.sampler)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        obs = {
            "base_image": np.moveaxis(sample["base_image"], -1, 1).astype(np.float32) / 255.0,
            "wrist_image": np.moveaxis(sample["wrist_image"], -1, 1).astype(np.float32) / 255.0,
            "robot_state": sample["robot_state"].astype(np.float32),
        }
        data = {
            "obs": obs,
            "action": sample["action"].astype(np.float32),
        }
        return dict_apply(data, torch.from_numpy)
