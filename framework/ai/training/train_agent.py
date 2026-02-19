from framework.analysis.technical_indicators import TechnicalIndicators
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback

from framework.ai.training.trading_env import CryptoTradingEnv
from framework.ai.models.transformer_feature_extractor import TransformerFeatureExtractor

# --- Configuration ---
DATA_PATH = "framework/data/BTC_USDT_5m_raw.csv"

WINDOW_SIZE = 60  # Look back 60 candles (5 hours)
TOTAL_TIMESTEPS = 1_000_000

# Raw Features to process
RAW_FEATURES = ["close", "volume", "rsi", "macd", "macd_signal", "macd_hist", "bb_upper", "bb_lower", "atr", "adx", "stochrsi_k", "stochrsi_d"]


class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """

    def __init__(self, verbose=0):
        super(TensorboardCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        if self.locals.get("infos"):
            info = self.locals["infos"][0]
            self.logger.record("custom/net_worth", info.get("net_worth"))
            self.logger.record("custom/position", info.get("pos"))
        return True


def make_env(df, features, rank, seed=0):
    def callback() -> CryptoTradingEnv:
        env = CryptoTradingEnv(
            df=df,
            features=features,
            window_size=WINDOW_SIZE,
            initial_balance=1000.0,
        )
        env.reset(seed=seed + rank)
        return env

    return callback


def train():
    print("--- Starting AI Training Session ---")

    # 1. Load Data
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Data file not found: {DATA_PATH}")

    print(f"Loading data from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)

    # 2. Split Data using sklearn (70% Train, 15% Val, 15% Test)
    # Important: shuffle=False to preserve time-series order
    print("Splitting data into Train (70%), Validation (15%), and Test (15%)...")

    # First split: 70% Train, 30% Temp (Val + Test)
    train_df, temp_df = train_test_split(df, test_size=0.3, shuffle=False)

    # Second split: Split the 30% Temp into 50% Val (15% total) and 50% Test (15% total)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, shuffle=False)

    # Reset indices to ensure environment works correctly
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    ti = TechnicalIndicators()

    ti.fit_scalers(train_df)

    train_df = ti.normalize(train_df)
    val_df = ti.normalize(val_df)
    test_df = ti.normalize(test_df)

    # 3. Create Environment (Training on Train Set)
    env = DummyVecEnv([make_env(train_df, train_df.columns, i) for i in range(1)])

    # 4. Initialize PPO
    print("Initializing PPO Agent...")

    policy_kwargs = dict(
        features_extractor_class=TransformerFeatureExtractor,
        features_extractor_kwargs=dict(features_dim=128),
    )

    model = PPO("MlpPolicy", env, verbose=1, policy_kwargs=policy_kwargs, learning_rate=3e-4, n_steps=2048, batch_size=64, gamma=0.99, gae_lambda=0.95, tensorboard_log=LOG_DIR, device="auto")

    # 5. Train
    print(f"Training for {TOTAL_TIMESTEPS} steps...")
    try:
        model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=TensorboardCallback())
    except KeyboardInterrupt:
        print("Training interrupted manually. Saving current model...")

    # 6. Save
    print(f"Saving model to {MODEL_SAVE_PATH}...")
    model.save(MODEL_SAVE_PATH)
    print("Done. (Note: Validation and Test simulation should be run separately using the saved model)")


if __name__ == "__main__":
    train()
