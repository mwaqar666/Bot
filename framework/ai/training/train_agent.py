import pandas as pd
from wandb import init as wandb_init, Run
from wandb.integration.sb3 import WandbCallback
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecEnv, DummyVecEnv
from stable_baselines3.common.callbacks import CallbackList

import config

from framework.ai.training.trading_env import TradingEnvironment
from framework.ai.models.tcn_feature_extractor import TCNFeatureExtractor


class ModelTrainer:
    def __init__(self) -> None:
        self.symbol = config.SYMBOL
        self.initial_balance = 10_000
        self.risk_per_trade = 0.01  # Risk 1% of balance per trade
        self.fee_percent = 0.001  # 0.1% maker/taker fee
        self.slippage_percent = 0.0005  # 0.05% slippage
        self.sl_multiplier = 1.0  # Stop Loss: 1x ATR
        self.tp_multiplier = 2.0  # Take Profit: 2x ATR (2:1 Risk/Reward)
        self.window_size = 60  # Look back 60 candles (5 hours)
        self.total_timesteps = 1_000_000

        self.model_path = "framework/ai/models/ppo_tcn_v1"
        self.wandb_project = "trading-bot"  # Name of your project on wandb.ai

    def train(self, train_df: pd.DataFrame, features: list[str]) -> None:
        print(f"Training on {len(features)} Features: {features}")

        # 1. Create Environment (Training on Train Set)
        env = self.__create_vector_environment_callback(train_df, features)

        # 2. Initialize PPO
        print("Initializing PPO Agent...")

        model = self.__create_model(env)

        # 3. Initialize W&B run
        run, callback = self.__initialize_weights_and_biases(features)

        callbacks = CallbackList([callback])

        # 4. Train
        print(f"Training for {self.total_timesteps} steps...")
        try:
            model.learn(total_timesteps=self.total_timesteps, callback=callbacks)
        except KeyboardInterrupt:
            print("Training interrupted manually. Saving current model...")
        finally:
            run.finish()

        # 5. Save
        print(f"Saving model to {self.model_path}...")
        model.save(self.model_path)
        print("Training complete.")

    def evaluate(self, df: pd.DataFrame, features: list[str], label: str = "Validation") -> dict:
        """
        Run the trained model on a dataset (val or test) and collect performance metrics.
        The model runs in deterministic mode (no exploration), so results are reproducible.

        Args:
            df (pd.DataFrame): The dataset to evaluate on (val_df or test_df).
            features (list[str]): List of feature column names fed to the model.
            label (str): Label for printing purposes (e.g. "Validation" or "Test").

        Returns:
            dict: A dictionary of performance metrics.
        """
        print(f"\n--- Evaluating on {label} Set ---")

        env = self.__create_environment(df, features)

        model = PPO.load(self.model_path, env=env)
        obs, _ = env.reset()
        done = False

        while not done:
            # deterministic=True: picks the highest-probability action, no random sampling
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

        # Collect metrics from trade history
        trades = env.trade_history
        wins = [t for t in trades if t > 0]
        losses = [t for t in trades if t <= 0]

        metrics = {
            "final_balance": env.balance,
            "total_return_%": (env.balance - self.initial_balance) / self.initial_balance * 100,
            "total_trades": len(trades),
            "win_rate_%": len(wins) / len(trades) * 100 if trades else 0.0,
            "avg_win_%": sum(wins) / len(wins) * 100 if wins else 0.0,
            "avg_loss_%": sum(losses) / len(losses) * 100 if losses else 0.0,
            "avg_trade_pnl_%": sum(trades) / len(trades) * 100 if trades else 0.0,
        }

        print(f"  Final Balance : ${metrics['final_balance']:.2f}  (Started: ${self.initial_balance:.2f})")
        print(f"  Total Return  : {metrics['total_return_%']:+.2f}%")
        print(f"  Total Trades  : {metrics['total_trades']}")
        print(f"  Win Rate      : {metrics['win_rate_%']:.1f}%")
        print(f"  Avg Win       : {metrics['avg_win_%']:+.3f}%")
        print(f"  Avg Loss      : {metrics['avg_loss_%']:+.3f}%")
        print(f"  Avg Trade PnL : {metrics['avg_trade_pnl_%']:+.3f}%")
        print("-----------------------------------")

        return metrics

    def __create_vector_environment_callback(self, df: pd.DataFrame, features: list[str]) -> DummyVecEnv:
        return DummyVecEnv([lambda: self.__create_environment(df, features)])

    def __create_environment(self, df: pd.DataFrame, features: list[str]) -> TradingEnvironment:
        return TradingEnvironment(
            df=df,
            symbol=self.symbol,
            window_size=self.window_size,
            features=features,
            initial_balance=self.initial_balance,
            risk_per_trade=self.risk_per_trade,
            fee_percent=self.fee_percent,
            slippage_percent=self.slippage_percent,
            sl_multiplier=self.sl_multiplier,
            tp_multiplier=self.tp_multiplier,
        )

    def __create_model(self, environment: VecEnv) -> PPO:
        policy_kwargs = dict(
            features_extractor_class=TCNFeatureExtractor,
            features_extractor_kwargs=dict(features_dim=64),
            net_arch=dict(pi=[32, 32], vf=[32, 32]),
        )

        return PPO(
            "MlpPolicy",
            environment,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            ent_coef=0.01,
            verbose=1,
            policy_kwargs=policy_kwargs,
        )

    def __initialize_weights_and_biases(self, features: list[str]) -> tuple[Run, WandbCallback]:
        run = wandb_init(
            project=self.wandb_project,
            config={
                "learning_rate": 1e-4,
                "n_steps": 2048,
                "batch_size": 64,
                "n_epochs": 10,
                "gamma": 0.99,
                "ent_coef": 0.01,
                "sl_multiplier": self.sl_multiplier,
                "tp_multiplier": self.tp_multiplier,
                "window_size": self.window_size,
                "n_features": len(features),
            },
            sync_tensorboard=False,
            save_code=True,
        )

        callbacks = WandbCallback(gradient_save_freq=500, verbose=2)

        return run, callbacks
