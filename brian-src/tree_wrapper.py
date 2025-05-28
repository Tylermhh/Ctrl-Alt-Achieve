import gym
import numpy as np
import torch as th
import torch.nn.functional as F
from gym import spaces


class MineRLTreeWrapper(gym.Wrapper):
    def __init__(self, env, camera_bins=360):
        super().__init__(env)

        # --- button list & camera bins ---
        self.buttons = [
            "attack","forward",
            "jump","left","right"
        ]
        self.camera_bins = camera_bins

        # Flat MultiDiscrete action space for model to use
        # Buttons are binary (0 for not pressed, 1 for pressed)
        # Camera has two axes each in range [0, camera_bins)
        self.action_space = spaces.MultiDiscrete(
            [2] * len(self.buttons) + [camera_bins, camera_bins]
        )

        # Flat observation space to pass to model
        inv_dim = len(self.env.observation_space['inventory'])
        self.observation_space = spaces.Dict({
            "image": spaces.Box(0.0, 1.0, shape=(1, 84, 84), dtype=np.float32),
            "inv":   spaces.Box(0.0, 100.0, shape=(inv_dim,), dtype=np.float32),
        })

        # Map inventory items to rewards
        self.reward_map = {
           'log':10, 'planks':20, 'stick':40
        }

    def reset(self, **kwargs):
        raw_obs = self.env.reset(**kwargs)
        self._last_raw_obs = raw_obs
        return self._process_obs(raw_obs)

    def step(self, action: np.ndarray):
        # Unpack flat MultiDiscrete → MineRL dict
        action_unflattened = self._unflatten_action(action)

        raw_obs, _, done, info = self.env.step(action_unflattened)
        self._last_raw_obs = raw_obs

        # Reformat obs
        obs = self._process_obs(raw_obs)

        # Compute shaped reward
        reward = self._shape_reward()

        # Encourage breaking and movement
        if action_unflattened["attack"] == 1:
            reward += 0.4

        if action_unflattened["forward"] == 1:
            reward += 0.1
        else:
            if action_unflattened["left"] == 1 and action_unflattened["right"] == 1:
                reward -= 0.4
            else:
                reward += 0.1

        if sum(action_unflattened["camera"] > 0):
            reward += 0.2

        return obs, reward, done, info

    def render(self, mode="human"):
        return self.env.render()

    def _unflatten_action(self, md: np.ndarray) -> dict:
        # First N entries: buttons
        n_btn = len(self.buttons)
        btn_vals = md[:n_btn].astype(int).flatten()
        # Last 2 entries: camera indices in [0, camera_bins)
        cam_idx = md[n_btn:].astype(int)

        # Ensure cam_idx has exactly 2 elements
        if cam_idx.shape[0] != 2:
            # Overwrite invalid camera actions
            cam = np.array([0.0, 0.0], dtype=np.float32)
        else:
            # Map index → angle in [-180,180]
            cam = (cam_idx / (self.camera_bins - 1)) * 360.0 - 180.0
            cam = cam.astype(np.float32)
            cam = np.clip(cam, -180.0, 180.0)
            cam = np.nan_to_num(cam, nan=0.0, posinf=0.0, neginf=0.0)

        # Rebuild MineRL action dict
        action_dict = {name: int(btn_vals[i]) for i, name in enumerate(self.buttons)}
        action_dict["camera"] = cam.astype(np.float32)
        return action_dict

    def _process_obs(self, obs):
        # Convert image to grayscale, 84x84
        img = obs['pov'].astype(np.float32) / 255.0
        t = th.tensor(img).permute(2,0,1)[None]  # (1,3,H,W)
        small = F.interpolate(t, (84,84), mode='bilinear').mean(1, keepdim=True)
        gray = small.numpy().astype(np.float32)    # (1,1,84,84)

        # Vectorize inventory
        inv = np.array(list(obs['inventory'].values()), dtype=np.float32)
        return {"image": gray, "inv": inv}

    def _shape_reward(self) -> float:
        # Clean inventory to contain only items that yield reward
        inv = {key: value for key, value in self._last_raw_obs['inventory'].items()
               if key in self.reward_map and value > 0}

        # Calculate reward
        reward = float(sum(self.reward_map[item]
                         for item in inv.keys()))

        return reward
