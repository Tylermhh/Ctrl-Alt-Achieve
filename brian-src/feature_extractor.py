from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch as th
import torch.nn as nn


class MineRLFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=256):
        super().__init__(observation_space, features_dim)
        # Vision branch
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, 8, stride=4), nn.ReLU(),
            nn.Conv2d(16, 32, 4, stride=2), nn.ReLU(),
            nn.Flatten()
        )
        # Compute CNN output size
        with th.no_grad():
            n_flatten = self.cnn(th.zeros(1, *observation_space['image'].shape)).shape[1]
        # Inventory branch
        inv_dim = observation_space['inv'].shape[0]
        self.inv_net = nn.Sequential(nn.Linear(inv_dim, 32), nn.ReLU())
        # Final MLP
        self.linear = nn.Sequential(
            nn.Linear(n_flatten + 32, features_dim),
            nn.ReLU()
        )

    def forward(self, observations):
        img_feat = self.cnn(observations["image"])
        inv_feat = self.inv_net(observations["inv"])
        return self.linear(th.cat([img_feat, inv_feat], dim=1))

