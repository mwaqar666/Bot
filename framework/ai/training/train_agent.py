import pandas as pd
from wandb import init as wandb_init, Run
from wandb.integration.sb3 import WandbCallback
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecEnv, DummyVecEnv
from stable_baselines3.common.callbacks import CallbackList, BaseCallback

import config

from framework.ai.training.trading_env import TradingEnvironment


class TradeMetricsCallback(BaseCallback):
    """
    Logs custom trading metrics from the environment's info dict into
    the SB3 logger (stdout table + W&B) on every rollout.
    """

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            self.logger.record("trade/net_worth", info.get("net_worth", 0))
            self.logger.record("trade/balance", info.get("balance", 0))
            self.logger.record("trade/trade_pnl", info.get("trade_pnl", 0))
            self.logger.record("trade/reward", info.get("reward", 0))
            self.logger.record("trade/action", info.get("action", 0))
        return True


class ModelTrainer:
    def __init__(
        self,
        features: list[str],
        symbol: str = config.SYMBOL,
        initial_balance: float = 10_000,
        risk_per_trade: float = 0.01,
        fee_percent: float = 0.001,
        slippage_percent: float = 0.0005,
        sl_multiplier: float = 1.0,
        tp_multiplier: float = 2.0,
        window_size: int = 1,
        total_timesteps: int = 5_000_000,
    ) -> None:
        self.features = features
        self.symbol = symbol
        self.initial_balance = initial_balance
        self.risk_per_trade = risk_per_trade
        self.fee_percent = fee_percent
        self.slippage_percent = slippage_percent
        self.sl_multiplier = sl_multiplier
        self.tp_multiplier = tp_multiplier
        self.window_size = window_size
        self.total_timesteps = total_timesteps

        self.learning_rate = 3e-4
        self.n_steps = 2048
        self.batch_size = 64
        self.n_epochs = 10
        self.gamma = 0.99
        self.gae_lambda = 0.95
        self.ent_coef = 0.01

        self.model_path = "framework/ai/models/ppo_tcn_v1"
        self.wandb_project = "trading-bot"  # Name of your project on wandb.ai

    def train(self, train_df: pd.DataFrame) -> None:
        print(f"Training on {len(self.features)} Features: {self.features}")

        # 1. Create Environment (Training on Train Set)
        env = self.__create_vector_environment_callback(train_df)

        # 2. Initialize PPO
        print("Initializing PPO Agent...")

        model = self.__create_model(env)

        # 3. Initialize W&B run
        run, callback_list = self.__initialize_monitoring_and_logging()

        # 4. Train
        print(f"Training for {self.total_timesteps} steps...")
        try:
            model.learn(total_timesteps=self.total_timesteps, callback=callback_list)
        except KeyboardInterrupt:
            print("Training interrupted manually. Saving current model...")
        finally:
            run.finish()

        # 5. Save
        print(f"Saving model to {self.model_path}...")
        model.save(self.model_path)
        print("Training complete.")

    def evaluate(self, df: pd.DataFrame, label: str = "Validation") -> dict:
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

        env = self.__create_environment(df)

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

    def __create_vector_environment_callback(self, df: pd.DataFrame) -> DummyVecEnv:
        return DummyVecEnv([lambda: self.__create_environment(df)])

    def __create_environment(self, df: pd.DataFrame) -> TradingEnvironment:
        return TradingEnvironment(
            df=df,
            symbol=self.symbol,
            window_size=self.window_size,
            features=self.features,
            initial_balance=self.initial_balance,
            risk_per_trade=self.risk_per_trade,
            fee_percent=self.fee_percent,
            slippage_percent=self.slippage_percent,
            sl_multiplier=self.sl_multiplier,
            tp_multiplier=self.tp_multiplier,
        )

    def __create_model(self, environment: VecEnv) -> PPO:
        # policy_kwargs = dict(
        #     features_extractor_class=TCNFeatureExtractor,
        #     features_extractor_kwargs=dict(features_dim=64),
        #     net_arch=dict(pi=[32, 32], vf=[32, 32]),
        # )

        return PPO(
            "MlpPolicy",
            environment,
            learning_rate=self.learning_rate,
            n_steps=self.n_steps,
            batch_size=self.batch_size,
            n_epochs=self.n_epochs,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            ent_coef=self.ent_coef,
            verbose=1,
            # policy_kwargs=policy_kwargs,
        )

    def __initialize_monitoring_and_logging(self) -> tuple[Run, CallbackList]:
        run = wandb_init(
            project=self.wandb_project,
            config={
                "learning_rate": self.learning_rate,
                "n_steps": self.n_steps,
                "batch_size": self.batch_size,
                "n_epochs": self.n_epochs,
                "gamma": self.gamma,
                "gae_lambda": self.gae_lambda,
                "ent_coef": self.ent_coef,
                "sl_multiplier": self.sl_multiplier,
                "tp_multiplier": self.tp_multiplier,
                "window_size": self.window_size,
                "n_features": len(self.features),
            },
            sync_tensorboard=False,
            save_code=True,
        )

        callbacks = CallbackList(
            [
                TradeMetricsCallback(),
                WandbCallback(gradient_save_freq=500, verbose=2),
            ]
        )

        return run, callbacks
