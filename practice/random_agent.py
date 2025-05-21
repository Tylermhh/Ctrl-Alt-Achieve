import gym
import minerl
import random
import numpy as np

env = gym.make("MineRLBasaltFindCave-v0")
obs = env.reset()

done = False
while not done:
    action = env.action_space.noop()
    action["ESC"] = 0

    # --- Movement: 70% chance to move ---
    if random.random() < 0.7:
        action["forward"] = int(random.random() < 0.5)
        action["back"] = int(random.random() < 0.1)
        action["left"] = int(random.random() < 0.3)
        action["right"] = int(random.random() < 0.3)
        action["jump"] = int(random.random() < 0.2)
    else:
        action["forward"] = 0
        action["back"] = 0
        action["left"] = 0
        action["right"] = 0
        action["jump"] = 0

    # --- Camera: 30% chance to move camera ---
    if random.random() < 0.3:
        pitch = np.random.uniform(-2, 2)  # up/down
        yaw = np.random.uniform(-5, 5)   # left/right
        action["camera"] = [pitch, yaw]
    else:
        action["camera"] = [0.0, 0.0]

    # Disable all other actions
    for k in [
        "attack", "use", "place", "inventory", "hotbar.1", "hotbar.2", "hotbar.3",
        "hotbar.4", "hotbar.5", "hotbar.6", "hotbar.7", "hotbar.8", "hotbar.9",
        "craft", "equip", "drop", "chat"
    ]:
        action[k] = 0

    obs, reward, done, _ = env.step(action)
    env.render()

env.close()
