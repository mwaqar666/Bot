import pandas as pd
import numpy as np
from IPython.display import clear_output
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecEnv, DummyVecEnv
from stable_baselines3.common.callbacks import CallbackList, BaseCallback

from framework.ai.training.trading_env import TradingEnvironment


class TradeMetricsCallback(BaseCallback):
    def __init__(self, plot_every: int = 2048) -> None:
        super().__init__()
        self.plot_every = plot_every
        self.timesteps: list[int] = []
        self.history_cols: list[str] = ["action", "reward", "balance", "trade_pnl", "entry_price", "exit_price", "sl_price", "tp_price"]
        self.history: dict[str, list] = {col: [] for col in self.history_cols}

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            self.timesteps.append(self.num_timesteps)

            for col in self.history_cols:
                value = info.get(col, 0)
                self.history[col].append(value)
                self.logger.record(f"trade/{col}", value)

        if self.num_timesteps % self.plot_every == 0:
            self._refresh_plot()

        return True

    def _refresh_plot(self) -> None:

        ts = np.array(self.timesteps)
        actions = np.array(self.history["action"])
        balance = np.array(self.history["balance"])
        trade_pnl = np.array(self.history["trade_pnl"])
        exit_prices = np.array(self.history["exit_price"])
        sl_prices = np.array(self.history["sl_price"])
        tp_prices = np.array(self.history["tp_price"])

        # Mask for actual trades (HOLD = 2 produces no trade)
        trade_mask = actions != 2
        wins = trade_pnl[trade_mask & (trade_pnl > 0)]
        losses = trade_pnl[trade_mask & (trade_pnl <= 0)]

        # Exit reason detection
        sl_hit = trade_mask & np.isclose(exit_prices, sl_prices, rtol=1e-6)
        tp_hit = trade_mask & np.isclose(exit_prices, tp_prices, rtol=1e-6)
        close_hit = trade_mask & ~sl_hit & ~tp_hit

        clear_output(wait=True)
        fig, axes = plt.subplots(3, 2, figsize=(16, 12))
        fig.suptitle(f"Training Dashboard — Step {self.num_timesteps:,} | Trades: {trade_mask.sum()} | Win Rate: {len(wins) / max(trade_mask.sum(), 1) * 100:.1f}%", fontsize=13)

        # 1. Balance over time
        axes[0, 0].plot(ts, balance, linewidth=0.8, color="steelblue")
        axes[0, 0].set_title("Balance ($)")
        axes[0, 0].set_ylabel("$")

        # 2. Per-trade PnL scatter (green=win, red=loss)
        colours = ["green" if p > 0 else "red" for p in trade_pnl[trade_mask]]
        axes[0, 1].scatter(ts[trade_mask], trade_pnl[trade_mask], s=2, alpha=0.4, c=colours)
        axes[0, 1].axhline(0, color="black", linewidth=0.8, linestyle="--")
        axes[0, 1].set_title("Trade PnL per Step (%)")

        # 3. Action distribution
        action_labels = ["BUY", "SELL", "HOLD"]
        action_counts = [(actions == i).sum() for i in range(3)]
        axes[1, 0].bar(action_labels, action_counts, color=["green", "red", "grey"])
        axes[1, 0].set_title("Action Distribution")
        axes[1, 0].set_ylabel("Count")

        # 4. Win / loss histogram
        axes[1, 1].hist(wins, bins=30, color="green", alpha=0.6, label=f"Wins ({len(wins)})")
        axes[1, 1].hist(losses, bins=30, color="red", alpha=0.6, label=f"Losses ({len(losses)})")
        axes[1, 1].axvline(0, color="black", linewidth=0.8)
        axes[1, 1].set_title("Win / Loss Distribution")
        axes[1, 1].set_xlabel("PnL (%)")
        axes[1, 1].legend()

        # 5. Exit reason (SL / TP / Closed at candle close)
        exit_counts = [sl_hit.sum(), tp_hit.sum(), close_hit.sum()]
        axes[2, 0].bar(["SL Hit", "TP Hit", "Candle Close"], exit_counts, color=["red", "green", "steelblue"])
        axes[2, 0].set_title("Exit Reason")
        axes[2, 0].set_ylabel("Count")

        # 6. Cumulative trade PnL
        cum_pnl = np.cumsum(trade_pnl[trade_mask])
        axes[2, 1].plot(range(len(cum_pnl)), cum_pnl, color="purple", linewidth=0.8)
        axes[2, 1].axhline(0, color="black", linewidth=0.8, linestyle="--")
        axes[2, 1].set_title("Cumulative Trade PnL")
        axes[2, 1].set_xlabel("Trade #")

        plt.tight_layout()
        plt.show()


class ModelTrainer:
    def __init__(
        self,
        features: list[str],
        model_save_path: str,
        initial_balance: float = 10_000,
        risk_per_trade: float = 0.01,
        fee_percent: float = 0.001,
        sl_multiplier: float = 1.0,
        tp_multiplier: float = 2.0,
        window_size: int = 1,
        total_timesteps: int = 1_000_000,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.01,
    ) -> None:
        self.features = features
        self.model_save_path = model_save_path
        self.initial_balance = initial_balance
        self.risk_per_trade = risk_per_trade
        self.fee_percent = fee_percent
        self.sl_multiplier = sl_multiplier
        self.tp_multiplier = tp_multiplier
        self.window_size = window_size
        self.total_timesteps = total_timesteps

        # Model hyperparameters
        self.learning_rate = learning_rate
        self.n_steps = n_steps
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.ent_coef = ent_coef

    def train(self, train_df: pd.DataFrame) -> None:
        print(f"Training on {len(self.features)} Features: {self.features}")

        # 1. Create Environment (Training on Train Set)
        env = DummyVecEnv([lambda: self.__create_environment(train_df)])

        # 2. Initialize PPO
        print("Initializing PPO Agent...")

        model = self.__create_model(env)

        # 3. Create TradeMetrics Callback
        callback_list = CallbackList([TradeMetricsCallback()])

        # 4. Train
        print(f"Training for {self.total_timesteps} steps...")
        try:
            model.learn(total_timesteps=self.total_timesteps, callback=callback_list)
        except KeyboardInterrupt:
            print("Training interrupted manually. Saving current model...")

        # 5. Save
        print(f"Saving model to {self.model_save_path}...")
        model.save(self.model_save_path)
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

        model = PPO.load(self.model_save_path, env=env)
        obs, _ = env.reset()
        done = False

        while not done:
            # deterministic=True: picks the highest-probability action, no random sampling
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

        # Collect metrics from trade history (SimulatedTrade objects)
        trades = env.trade_history
        wins = [t for t in trades if t.pnl_pct > 0]
        losses = [t for t in trades if t.pnl_pct <= 0]

        metrics = {
            "final_balance": env.balance,
            "total_return_%": (env.balance - self.initial_balance) / self.initial_balance * 100,
            "total_trades": len(trades),
            "win_rate_%": len(wins) / len(trades) * 100,
            "avg_win_%": sum(t.pnl_pct for t in wins) / len(wins) * 100,
            "avg_loss_%": sum(t.pnl_pct for t in losses) / len(losses) * 100,
            "avg_trade_pnl_%": sum(t.pnl_pct for t in trades) / len(trades) * 100,
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

    def __create_environment(self, df: pd.DataFrame) -> TradingEnvironment:
        return TradingEnvironment(
            df=df,
            window_size=self.window_size,
            features=self.features,
            initial_balance=self.initial_balance,
            risk_per_trade=self.risk_per_trade,
            fee_percent=self.fee_percent,
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
