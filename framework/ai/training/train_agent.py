import os

import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv
from data_engine.feature_engineer import FeatureEngineer
from rl_env.trading_env import CryptoTradingEnv
from ai_bot.models.transformer_policy import TransformerFeatureExtractor


def load_training_data() -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """
    Loads base 15m data and additional context dataframes (30m, 1h, 4h).
    Returns:
        Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]: Base DF and Context Dict.
    """
    base_path = "ai_bot/data_engine/BTC_USDT_USDT_15m.csv"
    if not os.path.exists(base_path):
        raise FileNotFoundError(f"Base data {base_path} not found.")

    print(f"Loading Base Data from {base_path}...")
    df = pd.read_csv(base_path, index_col="timestamp", parse_dates=True)

    context_files = {
        "30m": "ai_bot/data_engine/BTC_USDT_USDT_30m.csv",
        "1h": "ai_bot/data_engine/BTC_USDT_USDT_1h.csv",
        "4h": "ai_bot/data_engine/BTC_USDT_USDT_4h.csv",
    }

    additional_dfs = {}
    for name, path in context_files.items():
        if os.path.exists(path):
            print(f"Loading Context {name}...")
            additional_dfs[name] = pd.read_csv(path, index_col="timestamp", parse_dates=True)
        else:
            print(f"Warning: {name} context missing.")

    return df, additional_dfs


def prepare_features(df: pd.DataFrame, context_dfs: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, list[str]]:
    """
    Engines features using FeatureEngineer.
    Returns:
        Tuple[pd.DataFrame, List[str]]: Processed DF and list of feature columns.
    """
    print("Engaging Feature Engineering...")
    engineer = FeatureEngineer()
    df_processed = engineer.process_data(df, additional_dfs=context_dfs)
    df_processed.sort_index(inplace=True)

    all_features = engineer.get_state_columns()
    available_features = [f for f in all_features if f in df_processed.columns]

    print(f"Features Prepared: {len(available_features)}")

    # Debug: Save Processed Data
    debug_path = "ai_bot/data_engine/debug_features.csv"
    print(f"DEBUG: Saving processed features to {debug_path}...")
    df_processed.to_csv(debug_path)

    return df_processed, available_features


def create_training_env(df: pd.DataFrame, features: list[str], window_size: int = 60) -> VecEnv:
    """Creates the Vectorized Gym Environment."""
    print(f"Creating Environment (Window={window_size})...")
    return DummyVecEnv([lambda: CryptoTradingEnv(df, features=features, window_size=window_size)])


def initialize_ppo_agent(env: VecEnv) -> PPO:
    """Initializes the PPO Agent with Transformer Policy."""
    print("Initializing PPO with Transformer...")
    policy_kwargs = dict(
        features_extractor_class=TransformerFeatureExtractor,
        features_extractor_kwargs=dict(features_dim=128),
    )

    return PPO(
        "MlpPolicy",
        env,
        verbose=1,
        policy_kwargs=policy_kwargs,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        ent_coef=0.01,  # Force Exploration
        gamma=0.95,  # Discount Factor
        gae_lambda=0.9,
        tensorboard_log="./tensorboard_logs/",
    )


def train_model_orchestrator():
    """Main Orchestrator function."""
    print("--- Starting AI Training Session ---")

    try:
        # 1. Load Data
        df, context_dfs = load_training_data()

        # 2. Prepare Features
        df_processed, features = prepare_features(df, context_dfs)

        # 3. Create Env
        env = create_training_env(df_processed, features)

        # 4. Init Agent
        model = initialize_ppo_agent(env)
        print(model.policy)

        # 5. Train
        # Increase steps to allow Transformer to learn complex patterns (was 100k)
        print("Training for 250,000 steps...")
        model.learn(total_timesteps=250000)

        # 6. Save
        save_path = "ai_bot/models/ppo_transformer_bot"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        model.save(save_path)
        print(f"Model saved to {save_path}.zip")

    except Exception as e:
        print(f"Critical Error: {e}")


if __name__ == "__main__":
    train_model_orchestrator()
