import torch
from torch import Tensor, nn

import math 
from typing import Tuple, Type

from .common import MLPBlock


class TwoWayTransformer(nn.Module):
    def __init__(
            self, 
            depth: int, 
            embedding_dim: int, 
            num_heads: int,
            mlp_dim: int, 
            activation: Type[nn.Module] = nn.ReLU,
            attention_downsample_rate: int = 2,
            ) -> None:

