import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback

from framework.ai.training.trading_env import CryptoTradingEnv
from framework.ai.models.transformer_feature_extractor import TransformerFeatureExtractor

# --- Configuration ---
BASE_DIR = r"c:\Users\mwaqa\Desktop\Bot"
DATA_PATH = os.path.join(BASE_DIR, "framework", "data", "BTC_USDT_5m.csv")
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "framework", "ai", "models", "ppo_transformer_v1")
LOG_DIR = os.path.join(BASE_DIR, "framework", "ai", "logs")

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


def load_data(path: str) -> tuple[pd.DataFrame, list[str]]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file not found: {path}")

    print(f"Loading data from {path}...")
    df = pd.read_csv(path)

    # Validation
    missing = [f for f in RAW_FEATURES if f not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    # Create Normalized Features for the Agent
    # We leave original columns untouched for the Environment logic (PnL, Stop Loss)
    agent_features = []

    for col in RAW_FEATURES:
        norm_col_name = f"{col}_norm"
        # Robust Scaling (Median/IQR) or Z-Score (Mean/Std)
        # Using Z-Score for simplicity and neural net compatibility
        mean = df[col].mean()
        std = df[col].std()

        # Avoid division by zero
        if std == 0:
            std = 1

        df[norm_col_name] = (df[col] - mean) / std
        agent_features.append(norm_col_name)

    print("Data Normalized. Agent will see columns ending in '_norm'.")
    return df, agent_features


def make_env(df, features, rank, seed=0):
    """
    Utility function for multiprocessed env.
    """

    def _init():
        env = CryptoTradingEnv(
            df=df,
            features=features,  # Agent sees normalized features
            window_size=WINDOW_SIZE,
            initial_balance=1000.0,
        )
        env.reset(seed=seed + rank)
        return env

    return _init


def train():
    print("--- Starting AI Training Session ---")

    # 1. Load Data
    df, agent_features = load_data(DATA_PATH)
    print(f"Total Data Rows: {len(df)}")
    print(f"Agent Features: {agent_features}")

    # 2. Split Data using sklearn (70% Train, 15% Val, 15% Test)
    # Important: shuffle=False to preserve time-series order
    print("Splitting data into Train (70%), Validation (15%), and Test (15%)...")

    # First split: 70% Train, 30% Temp (Val + Test)
    train_df, temp_df = train_test_split(df, test_size=0.3, shuffle=False)

    # Second split: Split the 30% Temp into 50% Val (15% total) and 50% Test (15% total)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, shuffle=False)

    print(f"Train Set: {len(train_df)} rows")
    print(f"Val Set:   {len(val_df)} rows")
    print(f"Test Set:  {len(test_df)} rows")

    # Reset indices to ensure environment works correctly
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    # 3. Create Environment (Training on Train Set)
    env = DummyVecEnv([make_env(train_df, agent_features, i) for i in range(1)])

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
