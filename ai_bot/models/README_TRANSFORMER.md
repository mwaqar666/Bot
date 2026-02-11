# Transformer Policy Implementation Plan

## Objective
Implement a custom Feature Extractor for Stable Baselines 3 (SB3) that uses a Transformer architecture to process sequential market data. This allows the PPO agent to learn from time-series patterns (sequences of candles) rather than just static snapshots.

## Architecture

### 1. Input Processing
- **Input Shape**: `(Batch_Size, Window_Size, Num_Features)`
    - `Batch_Size`: Number of environments or samples in a mini-batch.
    - `Window_Size`: T steps of history (e.g., 60). *Note: Ensure `trading_env.py` provides this.*
    - `Num_Features`: D features per step (Close, RSI, etc.).
- **Transformation**: `Linear(Num_Features -> d_model)`
    - Projects the raw features into a higher-dimensional embedding space.
    - `d_model`: 64 (Embedding Dimension).

### 2. Positional Encoding
- **Purpose**: Transformers are permutation-invariant. We must inject information about the order of the sequence.
- **Method**: Learnable Positional Embedding.
    - `self.pos_encoder = nn.Parameter(torch.randn(1, Window_Size, d_model))`
    - added to the embedded input: `x = x + self.pos_encoder`

### 3. Transformer Encoder
- **Core Component**: `nn.TransformerEncoder`
- **Configuration**:
    - `d_model`: 64
    - `nhead`: 4 (Number of Attention Heads - allows attending to different feature subspaces)
    - `dim_feedforward`: 128 (Internal dimension of the FFN inside the transformer)
    - `dropout`: 0.1
    - `batch_first`: `True` (Crucial for SB3 compatibility)
    - `num_layers`: 2 (Depth of the network)

### 4. Output Projection
- **Flattening**: The transformer outputs a sequence `(Batch, Window, d_model)`. We flatten this to `(Batch, Window * d_model)`.
- **Final Layer**: `Linear(Window * d_model -> features_dim)`
    - Projects the flattened representation to the requested `features_dim` (e.g., 128, 256).
    - This vector is what the PPO Actor/Critic heads will consume.

## Implementation Details (File: `ai_bot/models/transformer_policy.py`)

### Dependencies
- `torch`
- `torch.nn`
- `gymnasium`
- `stable_baselines3.common.torch_layers.BaseFeaturesExtractor`

### Class Structure
```python
class TransformerFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        ...
        
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        ...
```

### Key Considerations
1.  **Observation Space Validation**: Ensure `observation_space.shape` is 2D `(Window, Features)`. If it's 1D, the environment is configured wrong or `Flatten` wrapper is being used inadvertently.
2.  **Device Compatibility**: Use `torch.device` awareness if needed (SB3 handles this mostly, but good to be safe).
3.  **Hyperparameters**:
    - `d_model=64`: Balanced for financial time series (not too complex).
    - `nhead=4`: Enough heads to capture different market dynamics (trend, volatility, mean reversion, volume).
    - `layers=2`: Shallow enough to train fast, deep enough for basic reasoning.

## Checklist
- [ ] Create `ai_bot/models/transformer_policy.py`.
- [ ] Implement `TransformerFeatureExtractor` class.
- [ ] Verify `forward` pass shape transformations.
- [ ] Integrate into `train_agent.py` (pass via `policy_kwargs`).
