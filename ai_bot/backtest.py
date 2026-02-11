import numpy as np
import sys
import os
import pandas as pd
from typing import Tuple, List

# Add parent path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from ai_bot.data_engine.feature_engineer import FeatureEngineer
from ai_bot.rl_env.trading_env import CryptoTradingEnv
from ai_bot.models.transformer_policy import TransformerFeatureExtractor

# Monkey Patch for NumPy 2.0 -> 1.x Compatibility
try:
    import numpy.core

    # 1. Provide numpy._core if missing
    if "numpy._core" not in sys.modules:
        sys.modules["numpy._core"] = numpy.core
        np._core = numpy.core

    # 2. Provide numpy._core.numeric if missing (Critical for pickle)
    if "numpy._core.numeric" not in sys.modules:
        sys.modules["numpy._core.numeric"] = numpy.core.numeric
        np._core.numeric = numpy.core.numeric

except Exception as e:
    print(f"Warning: NumPy Patch failed: {e}")

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv


# --- Configuration for Backtest ---
MODEL_PATH = "ai_bot/models/ppo_transformer_bot"
DATA_PATH = "ai_bot/data_engine/BTC_USDT_USDT_15m.csv"
WINDOW_SIZE = 60


def load_and_process_data(csv_path: str) -> Tuple[pd.DataFrame, List[str]]:
    """Loads CSV data and applies Feature Engineering with HTF Context."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Data file not found at {csv_path}")

    print(f"Loading Base Data from {csv_path}...")
    df = pd.read_csv(csv_path)

    # Ensure timestamp is parsed
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df.set_index("timestamp", inplace=True)

    # --- Load Higher Timeframe Data ---
    base_dir = os.path.dirname(csv_path)
    # Mapping based on standard conventions
    htf_files = {
        "30m": "BTC_USDT_USDT_30m.csv",
        "1h": "BTC_USDT_USDT_1h.csv",
        "4h": "BTC_USDT_USDT_4h.csv",
    }

    additional_dfs = {}
    print("Loading Higher Timeframe Context...")
    for tf, filename in htf_files.items():
        full_path = os.path.join(base_dir, filename)
        if os.path.exists(full_path):
            print(f"  - Loading {tf}: {filename}")
            htf_df = pd.read_csv(full_path)
            if "timestamp" in htf_df.columns:
                htf_df["timestamp"] = pd.to_datetime(htf_df["timestamp"])
                htf_df.set_index("timestamp", inplace=True)
            additional_dfs[tf] = htf_df
        else:
            print(f"  [WARNING] Missing {tf} context file: {filename}")

    print("Applying Feature Engineering...")
    fe = FeatureEngineer()
    # Pass additional_dfs to generate HTF features
    df = fe.process_data(df, additional_dfs=additional_dfs)

    # Ensure 'timestamp' column exists for Env if it was dropped or index used
    if "timestamp" not in df.columns:
        df["timestamp"] = df.index

    return df, fe.get_state_columns()


def setup_backtest_env(df: pd.DataFrame, features: List[str]) -> DummyVecEnv:
    """Initializes the Trading Environment."""
    return DummyVecEnv(
        [lambda: CryptoTradingEnv(df, features=features, window_size=WINDOW_SIZE)]
    )


def load_agent(env: DummyVecEnv, model_path: str) -> PPO:
    """Loads the trained PPO Agent with custom policy objects."""
    print(f"Loading model from {model_path}...")
    try:
        custom_objects = {
            "features_extractor_class": TransformerFeatureExtractor,
            "features_extractor_kwargs": dict(features_dim=128),
        }
        return PPO.load(model_path, env=env, custom_objects=custom_objects)
    except Exception as e:
        print(f"Error loading model: {e}")
        # Fallback: try loading just the policy if env mismatch (though likely fails)
        return PPO.load(model_path, custom_objects=custom_objects)


def run_simulation(env: DummyVecEnv, model: PPO, total_steps: int):
    """Executes the simulation loop."""
    print(f"Running Backtest Simulation for {total_steps} steps...")
    obs = env.reset()

    for _ in range(total_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, dones, _ = env.step(action)

        if dones[0]:
            print("Episode Finished (End of Data).")
            break


def analyze_performance(env: DummyVecEnv, df: pd.DataFrame):
    """Generates performance report and saves trade history."""
    real_env = env.envs[0]
    trades = real_env.trade_history

    print("\n--- Account Status ---")
    print(f"Final Balance:   {real_env.balance:.2f} USDT")
    print(f"Final Net Worth: {real_env.net_worth:.2f} USDT")
    roi = (
        (real_env.net_worth - real_env.initial_balance) / real_env.initial_balance
    ) * 100
    print(f"Total Return:    {roi:.2f}%")

    if not trades:
        print("\n[WARNING] No trades were executed.")
        return

    _save_trade_history(trades, df)


def _save_trade_history(trades: list, df: pd.DataFrame):
    """Helper to process and save trade history to CSV."""
    trades_df = pd.DataFrame(trades)

    # Calculate Entry Step
    trades_df["entry_step"] = trades_df["step"] - trades_df["duration"]

    # Map Steps to Timestamps (Vectorized)
    entry_indices = trades_df["entry_step"].astype(int).values
    exit_indices = trades_df["step"].astype(int).values

    trades_df["entry_time"] = df.iloc[entry_indices]["timestamp"].values
    trades_df["exit_time"] = df.iloc[exit_indices]["timestamp"].values

    # Save
    output_csv = "backtest_trades.csv"
    trades_df.to_csv(output_csv, index=False)
    print(f"\n[SUCCESS] Saved {len(trades_df)} trades to {output_csv}")

    # Print Quick Stats
    _print_trade_stats(trades_df)


def _print_trade_stats(df: pd.DataFrame):
    """Prints summary statistics for trades."""
    total = len(df)
    wins = len(df[df["pnl"] > 0])
    win_rate = (wins / total) * 100
    total_pnl = df["pnl"].sum()
    avg_pnl = df["pnl"].mean()

    print("\n--- Trade Performance ---")
    print(f"Total Trades: {total}")
    print(f"Win Rate:     {win_rate:.2f}%")
    print(f"Total PnL:    {total_pnl:.2f} USDT")
    print(f"Avg PnL:      {avg_pnl:.2f} USDT")

    print("\n--- Recent Activity ---")
    cols = ["exit_time", "reason", "pnl", "entry_price", "exit_price"]
    print(df.tail(5)[cols].to_string(index=False))


def main():
    """Main execution flow."""
    try:
        # Pipeline
        df, features = load_and_process_data(DATA_PATH)
        env = setup_backtest_env(df, features)
        model = load_agent(env, MODEL_PATH)

        # Calculate steps (approximate total length minus window)
        total_steps = len(df) - WINDOW_SIZE

        run_simulation(env, model, total_steps)
        analyze_performance(env, df)

    except Exception as e:
        print(f"\n[CRITICAL ERROR] Backtest Failed: {e}")


if __name__ == "__main__":
    main()
