import torch as th
import numpy as np
from torch import nn
from psltl.rl_agents.common.torch_layers import BaseFeaturesExtractor
import gym


class Embedding(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space: gym.Space,
        embedding_dim: int = 500, 
        features_dim: int = 10,
    ) -> None:
        super().__init__(observation_space, features_dim)
        
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(embedding_dim, features_dim)

    def forward(self, observations: th.Tensor) -> th.Tensor:
        device = observations.device
        observations = th.squeeze(observations.long())

        observations = observations.cpu().long()
        idxes = np.where(observations == 1)
        idx = idxes[0]
        output = self.embedding(th.Tensor(idx).to(device).long())
        
        return output
