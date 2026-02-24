import torch
import torch.nn as nn
import gymnasium as gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class TCNFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int) -> None:
        super(TCNFeatureExtractor, self).__init__(observation_space, features_dim)

        # input_shape: (Window, Features) -> (60, 31)
        self.window_size, input_channels = observation_space.shape

        # ======================================================================================
        # TCN Architecture. We use Dilated Convolutions to expand the receptive field
        # ======================================================================================
        # Receptive Field = 1 + (Kernel Size - 1) * Sum(Dilation)
        # Receptive Field = 1 + (3 - 1) * (1 + 2 + 4 + 8 + 16) = 1 + 2 * 31 = 63 <- This covers our 60 candle window
        # ======================================================================================
        # L_out = ceil(((L_in + (2 * padding) - dilation * (kernel_size - 1) - 1) / stride) + 1)
        # ======================================================================================

        self.tcn = nn.Sequential(
            # Layer 1: Dilation 1 (Sees 3 candles)
            nn.Conv1d(input_channels, 64, kernel_size=3, padding=1, dilation=1),
            nn.LeakyReLU(0.1),
            # Layer 2: Dilation 2 (Sees 7 candles)
            nn.Conv1d(64, 128, kernel_size=3, padding=2, dilation=2),
            nn.LeakyReLU(0.1),
            # Layer 3: Dilation 4 (Sees 15 candles)
            nn.Conv1d(128, 128, kernel_size=3, padding=4, dilation=4),
            nn.LeakyReLU(0.1),
            # Layer 4: Dilation 8 (Sees 31 candles)
            nn.Conv1d(128, 64, kernel_size=3, padding=8, dilation=8),
            nn.LeakyReLU(0.1),
            # Layer 5: Dilation 16 (Sees 63 candles)
            nn.Conv1d(64, 32, kernel_size=3, padding=16, dilation=16),
            nn.LeakyReLU(0.1),
            nn.Flatten(),
        )

        # Compute the output shape of the TCN to feed into the final linear layer
        with torch.no_grad():
            sample_input = torch.randn(1, input_channels, self.window_size)
            _, n_flatten = self.tcn(sample_input).shape

        self.linear = nn.Linear(n_flatten, features_dim)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # SB3 gives: (Batch, Window, Features)
        # TCN needs: (Batch, Features, Window)
        x = observations.permute(0, 2, 1)
        x = self.tcn(x)
        return torch.relu(self.linear(x))
