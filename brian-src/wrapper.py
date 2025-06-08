import gym
import numpy as np
import torch
import torch.nn.functional as F
from gym import spaces


class MineRLWrapper(gym.Wrapper):
    def __init__(self, env, camera_bins=360):
        super().__init__(env)

        # Button list and discrete camera bins for actions
        self.buttons = [
            "ESC","attack","back","drop","forward",
            "hotbar.1","hotbar.2","hotbar.3","hotbar.4","hotbar.5",
            "hotbar.6","hotbar.7","hotbar.8","hotbar.9",
            "inventory","jump","left","pickItem","right",
            "sneak","sprint","swapHands","use"
        ]
        self.camera_bins = camera_bins

        # Flat MultiDiscrete action space for model to use
        # Buttons are binary (0 for not pressed, 1 for pressed)
        # Camera has two axes each in range [0, camera_bins)
        self.action_space = spaces.MultiDiscrete(
            [2] * len(self.buttons) + [camera_bins, camera_bins]
        )

        # Flattened observation space to pass to model
        inv_dim = len(self.env.observation_space['inventory'])
        self.observation_space = spaces.Dict({
            "image": spaces.Box(0.0, 1.0, shape=(1, 84, 84), dtype=np.float32),
            "inv":   spaces.Box(0.0, 100.0, shape=(inv_dim,), dtype=np.float32),
        })

        # Map inventory items to rewards
        self.reward_map = {
           'log':1, 'planks':2, 'stick':4, 'crafting_table':4,
           'wooden_pickaxe':8, 'cobblestone':16,
           'furnace':32, 'stone_pickaxe':32,
           'iron_ore':64, 'iron_ingot':128,
           'iron_pickaxe':256, 'diamond':1024,
           'diamond_shovel':2048
        }

    def reset(self, **kwargs):
        raw_obs = self.env.reset(**kwargs)
        self._last_raw_obs = raw_obs
        return self._process_obs(raw_obs)

    def step(self, action: np.ndarray):
        # Unpack flat MultiDiscrete into MineRL dict
        d = self._unflatten_action(action)

        raw_obs, _, done, info = self.env.step(d)
        self._last_raw_obs = raw_obs

        # Reformat obs
        obs = self._process_obs(raw_obs)

        # Compute shaped reward
        reward = self._shape_reward()

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
            # Map index to angle in [-180,180]
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
        t = torch.tensor(img).permute(2,0,1)[None]  # (1,3,H,W)
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
