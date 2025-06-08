import gym
import minerl # this will look unused but it is needed to fetch the environment!
from stable_baselines3 import PPO
from wrapper import MineRLWrapper
from feature_extractor import MineRLFeatureExtractor
from sys import argv
import os.path


# Create wrapped env
def make_minerl_env():
    raw = gym.make("MineRLObtainDiamondShovel-v0")
    return MineRLWrapper(raw)

env = make_minerl_env()
print("Succesfully created env")


def train(model_name: str, time_steps: int=2_000_000) -> None:
    """
    Train a model using custom wrapper and feature extractor.

    :param str model_name: File name to which to save the model
    :param int time_steps: Number of time steps to run training, default: 2_000_000
    """
    if os.path.exists(model_name+".zip"):
        # Load model
        model = PPO.load(model_name, env=env)
        print(f'============Loaded pre-existing model "{model_name}"============')
    else:
        # Define policy with custom feature extractor
        policy_kwargs = dict(
            features_extractor_class=MineRLFeatureExtractor,
            features_extractor_kwargs=dict(features_dim=256),
        )

        # Define model with custom env and policy
        model = PPO(
            "MultiInputPolicy",
            env,
            policy_kwargs=policy_kwargs,
            verbose=1,
            learning_rate=2.5e-4,
            n_steps=256,
            batch_size=256,
            n_epochs=4,
        )
        print(f'============Created new model============')


    # Train model
    print(f"============Beginning training with {time_steps} time steps============")
    model.learn(total_timesteps=time_steps)
    print("============Training complete============")
    model.save(model_name)
    print("============Saved model============")


def inference(model_name: str, time_steps: int=500) -> None:
    """
    Run inference using a model that is already trained

    :param str model_name: Name of the file from which to load the model
    :param int time_steps: Number of time steps to run inference, default: 500
    """
    # Load model
    model = PPO.load(model_name, env=env)

    # Evaluation stats
    total_reward = 0
    total_steps = 0
    current_inventory = {}

    obs = env.reset()

    print(f"============Beginning inference with {time_steps} time steps============")
    for _ in range(time_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)

        # Accumulate stats
        total_reward += reward
        total_steps += 1

        # Track inventory changes
        if "inventory" in obs:
            for item, count in obs["inventory"].items():
                current_inventory[item] = int(count)

        # Exit if there is an error
        if "error" in info:
            print("Step failed:", info["error"])
            break

        env.render()

        if done:
            print("done")
            obs = env.reset()

    # Print results
    print("============Inference complete============")
    print(f"Total reward: {total_reward}")
    print(f"Steps taken: {total_steps}")
    print("Inventory collected:")
    for item, count in current_inventory.items():
        if count > 0:
            print(f"  {item}: {count}")

if __name__ == "__main__":
    if len(argv) < 4 or (len(argv) >= 4 and not (argv[1] in ['T', 'I'] and argv[2].isnumeric())):
        print("Usage: python agent.py <T | I> <time steps> <model name>")
        exit()

    time_steps = int(argv[2])
    
    model_name = argv[3]

    if argv[1] == 'T':
        train(model_name, time_steps)
    elif argv[1] == 'I':
        inference(model_name, time_steps)
