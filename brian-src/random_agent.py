import gym
import minerl # This will look unused but it is needed to fetch the environment!
import logging

logging.basicConfig(level=logging.DEBUG)

env = gym.make("MineRLObtainDiamondShovel-v0")

obs = env.reset()

done = False

while not done:
    # Take a random action
    # action = env.action_space.sample()
    action = {"forward": 1, "jump": 1}
    # In BASALT environments, sending ESC action will end the episode
    # Lets not do that
    action["ESC"] = 0
    obs, reward, done, _ = env.step(action)
    print(obs)
    print(reward)
    env.render()

