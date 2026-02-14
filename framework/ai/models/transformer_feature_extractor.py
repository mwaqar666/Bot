import torch
import torch.nn as nn
import gymnasium as gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class TransformerFeatureExtractor(BaseFeaturesExtractor):
    """
    Custom Feature Extractor for Stable Baselines 3 using a Transformer Encoder.

    Architecture:
    Input (Window, Features) -> Linear Projection (d_model) -> Positional Encoding
    -> Transformer Encoder (N layers) -> Flatten -> Output Vector (features_dim)
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 128):
        """
        Args:
            observation_space (gym.spaces.Box): The observation space of the environment.
                                                Expected shape: (window_size, num_features).
            features_dim (int): The dimension of the output feature vector.
        """
        # Initialize the Base Class
        super(TransformerFeatureExtractor, self).__init__(
            observation_space, features_dim
        )

        # 1. Inspect Input Shape
        # We expect (Window_Size, Num_Features)
        # If the environment flattens it, we might need to reshape.
        # However, our custom env provides (Window, Features).
        self.window_size = observation_space.shape[0]
        self.input_dim = observation_space.shape[1]

        # Hyperparameters for the Transformer
        d_model = 64  # Embedding dimension
        nhead = 4  # Number of attention heads
        num_layers = 2  # Number of transformer layers
        dim_feedforward = 128  # Internal FFN dimension
        dropout = 0.1

        # 2. Input Projection Layer
        # Maps raw features (e.g. 15) to d_model (64)
        self.linear_in = nn.Linear(self.input_dim, d_model)

        # 3. Positional Encoding
        # Learnable parameter to give sequence order information
        self.pos_encoder = nn.Parameter(torch.randn(1, self.window_size, d_model))

        # 4. Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,  # Important: SB3 uses (Batch, Seq, Feat)
            activation="relu",
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        # 5. Output Projection
        # Flatten: (Batch, Window, d_model) -> (Batch, Window * d_model)
        self.flatten = nn.Flatten()

        # Final Linear Layer to map to features_dim
        self.linear_out = nn.Linear(d_model * self.window_size, features_dim)

        # Activation function
        self.relu = nn.ReLU()

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the network.

        Args:
            observations (torch.Tensor): Input batch of observations.
                                         Shape: (Batch_Size, Window_Size, Num_Features)

        Returns:
            torch.Tensor: Feature vector of shape (Batch_Size, features_dim)
        """
        # 1. Project Input
        # x shape: (Batch, Window, d_model)
        x = self.linear_in(observations)

        # 2. Add Positional Encoding
        # Broadcasting adds (1, Window, d_model) to (Batch, Window, d_model)
        x = x + self.pos_encoder

        # 3. Pass through Transformer
        # x shape: (Batch, Window, d_model)
        x = self.transformer_encoder(x)

        # 4. Flatten
        # x shape: (Batch, Window * d_model)
        x = self.flatten(x)

        # 5. Output Projection
        # x shape: (Batch, features_dim)
        x = self.relu(self.linear_out(x))

        return x
