import os

import config

from framework.data_engine.feature_engineer import FeatureEngineer
from framework.ai.training.trading_env import CryptoTradingEnv
from framework.models.transformer_policy import TransformerFeatureExtractor

import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv


class Backtester:
    """
    Handles backtesting of the trading agent.
    """

    def __init__(
        self,
        symbol: str = config.SYMBOL,
        timeframe: str = config.TIMEFRAME,
        model_path: str = "ai_bot/models/ppo_transformer_bot",
        data_dir: str = "ai_bot/data_engine/data",
        window_size: int = 60,
    ):
        self.symbol = symbol
        self.timeframe = timeframe
        self.model_path = model_path
        self.data_dir = data_dir
        self.window_size = window_size

    def run(self):
        """
        Main execution flow.
        """
        try:
            # Pipeline
            csv_path = self._get_data_path()
            df, features = self.load_and_process_data(csv_path)

            env = self.setup_backtest_env(df, features)
            model = self.load_agent(env)

            # Calculate steps (approximate total length minus window)
            total_steps = len(df) - self.window_size

            self.run_simulation(env, model, total_steps)
            self.analyze_performance(env, df)

        except Exception as e:
            print(f"\n[CRITICAL ERROR] Backtest Failed: {e}")

    def _get_data_path(self) -> str:
        """Constructs the path to the CSV data file."""
        safe_symbol = self.symbol.replace("/", "_").replace(":", "_")
        filename = f"{safe_symbol}_{self.timeframe}.csv"
        return os.path.join(self.data_dir, filename)

    def load_and_process_data(self, csv_path: str) -> tuple[pd.DataFrame, list[str]]:
        """Loads CSV data and applies Feature Engineering."""
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Data file not found at {csv_path}")

        print(f"Loading Base Data from {csv_path}...")
        df = pd.read_csv(csv_path)

        # Ensure timestamp is parsed
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df.set_index("timestamp", inplace=True)

        print("Applying Feature Engineering...")
        fe = FeatureEngineer()
        # Note: HTF context removed as FeatureEngineer logic doesn't currently support it
        df = fe.process_data(df)

        # Ensure 'timestamp' column exists for Env if it was dropped or index used
        if "timestamp" not in df.columns:
            df["timestamp"] = df.index

        # Filter features to only those that exist in the DataFrame
        # FeatureEngineer might define expected features that are not yet calculated
        expected_features = fe.get_state_columns()
        valid_features = [f for f in expected_features if f in df.columns]

        if len(valid_features) < len(expected_features):
            missing = set(expected_features) - set(valid_features)
            print(f"[WARNING] Missing features (not in Data): {missing}")

        return df, valid_features

    def setup_backtest_env(self, df: pd.DataFrame, features: list[str]) -> DummyVecEnv:
        """Initializes the Trading Environment."""
        return DummyVecEnv([lambda: CryptoTradingEnv(df, features=features, window_size=self.window_size)])

    def load_agent(self, env: DummyVecEnv) -> PPO:
        """Loads the trained PPO Agent with custom policy objects."""
        print(f"Loading model from {self.model_path}...")
        try:
            custom_objects = {
                "features_extractor_class": TransformerFeatureExtractor,
                "features_extractor_kwargs": dict(features_dim=128),
            }
            return PPO.load(self.model_path, env=env, custom_objects=custom_objects)
        except Exception as e:
            print(f"Error loading model: {e}")
            # Fallback
            return PPO.load(self.model_path, custom_objects=custom_objects)

    def run_simulation(self, env: DummyVecEnv, model: PPO, total_steps: int):
        """Executes the simulation loop."""
        print(f"Running Backtest Simulation for {total_steps} steps...")
        obs = env.reset()

        for _ in range(total_steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, _, dones, _ = env.step(action)

            if dones[0]:
                print("Episode Finished (End of Data).")
                break

    def analyze_performance(self, env: DummyVecEnv, df: pd.DataFrame):
        """Generates performance report and saves trade history."""
        real_env = env.envs[0]
        trades = real_env.trade_history

        print("\n--- Account Status ---")
        print(f"Final Balance:   {real_env.balance:.2f} USDT")
        print(f"Final Net Worth: {real_env.net_worth:.2f} USDT")
        roi = ((real_env.net_worth - real_env.initial_balance) / real_env.initial_balance) * 100
        print(f"Total Return:    {roi:.2f}%")

        if not trades:
            print("\n[WARNING] No trades were executed.")
            return

        self._save_trade_history(trades, df)

    def _save_trade_history(self, trades: list, df: pd.DataFrame):
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
        self._print_trade_stats(trades_df)

    def _print_trade_stats(self, df: pd.DataFrame):
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


if __name__ == "__main__":
    backtester = Backtester()
    backtester.run()
