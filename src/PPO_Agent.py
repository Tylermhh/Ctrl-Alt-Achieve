import gym
import numpy as np
import minerl
from stable_baselines3 import PPO

# Simple environment wrapper to deal with complex action spaces
# Custom Discretizer Wrapper
class DiscretizerWrapper(gym.Env):
    def __init__(self, env):
        super().__init__()
        self.env = env

        # Define a discrete action space of size N
        self.action_space = gym.spaces.Discrete(6)

        # Match the original observation space
        self.observation_space = env.observation_space

        # Default (no-op) action template
        self.default_action = {
            "ESC": 0, "attack": 0, "back": 0, "camera": np.array([0.0, 0.0], dtype=np.float32),
            "drop": 0, "forward": 0, "hotbar.1": 0, "hotbar.2": 0, "hotbar.3": 0, "hotbar.4": 0,
            "hotbar.5": 0, "hotbar.6": 0, "hotbar.7": 0, "hotbar.8": 0, "hotbar.9": 0,
            "inventory": 0, "jump": 0, "left": 0, "pickItem": 0, "right": 0,
            "sneak": 0, "sprint": 0, "swapHands": 0, "use": 0
        }

        # Map of discrete action index to full action dict
        self.action_map = {
            0: {**self.default_action, "forward": 1},
            1: {**self.default_action, "jump": 1},
            2: {**self.default_action, "attack": 1},
            3: {**self.default_action, "right": 1},
            4: {**self.default_action, "left": 1},
            5: {**self.default_action, "camera": np.array([15.0, 0.0], dtype=np.float32)},  # look right
        }

    def reset(self):
        return self.env.reset()
    
    def render(self, mode="human"):
        return self.env.render()

    def step(self, action):
        action_dict = self.action_map[int(action)]
        return self.env.step(action_dict)

# Create and wrap the environment
env = gym.make("MineRLBasaltFindCave-v0")
env = DiscretizerWrapper(env)  # Apply the discretizer wrapper

# # Initialize the PPO model
# print("-------- Initializing Model --------")
# model = PPO("MultiInputPolicy", env, n_steps=512, verbose=1)    # multiInputPolicy instead of CnnPolicysince we are working with dict observation space and not single rgb frames as the observations

# # Train the agent
# print("-------- Training Model --------")
# model.learn(total_timesteps=80000)

# # Save the trained model
# print("-------- Model Trained Successfully! Saving... --------")
# model.save("ppo_minerl_findcave")

# Load the model (just in case we wanna try loading in pre-trained ones)
print("-------- Loading trained model -------")
model = PPO.load("ppo_minerl_findcave")

# Use the trained model to interact with the environment
print("-------- Testing model in environment --------")
obs = env.reset()
done = False
while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render()



