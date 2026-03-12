import pandas as pd
import numpy as np
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecEnv, DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import CallbackList, BaseCallback
from stable_baselines3.common.logger import TensorBoardOutputFormat, configure

from framework.data.data_types import SignalDirection
from framework.ai.training.trading_env import TradingEnvironment


class TradeMetricsCallback(BaseCallback):
    def __init__(self, plot_every: int = 2048) -> None:
        super().__init__()
        self.plot_every = plot_every
        self.timesteps: list[int] = []

        self.tb_writer: SummaryWriter | None = None
        self.history_cols: list[str] = ["action", "balance", "trade_pnl", "entry_price", "exit_price", "sl_price", "tp_price"]
        self.history: dict[str, list] = {col: [] for col in self.history_cols}

    def _on_training_start(self) -> None:
        # Discover the TensorBoard writer created by SB3 (if tensorboard_log is enabled).
        for output_format in self.logger.output_formats:
            if isinstance(output_format, TensorBoardOutputFormat):
                self.tb_writer = output_format.writer
                break

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            self.timesteps.append(self.num_timesteps)

            for col in self.history_cols:
                value = info.get(col, 0)
                self.history[col].append(value)
                self.logger.record(f"trade/{col}", value)

        if self.num_timesteps % self.plot_every == 0:
            self._log_tensorboard_dashboard()

        return True

    def _on_training_end(self) -> None:
        # Ensure we always log the final dashboard snapshot even if
        # total_timesteps is not a multiple of plot_every.
        self._log_tensorboard_dashboard()

    def _log_tensorboard_dashboard(self) -> None:
        if self.tb_writer is None:
            return

        actions = np.array(self.history["action"])
        balance = np.array(self.history["balance"])
        trade_pnl = np.array(self.history["trade_pnl"])
        exit_prices = np.array(self.history["exit_price"])
        sl_prices = np.array(self.history["sl_price"])
        tp_prices = np.array(self.history["tp_price"])

        # Mask for actual trades (HOLD = 2 produces no trade)
        trade_mask = actions != SignalDirection.HOLD
        wins = trade_pnl[trade_mask & (trade_pnl > 0)]
        losses = trade_pnl[trade_mask & (trade_pnl <= 0)]

        # Exit reason detection
        sl_hit = trade_mask & np.isclose(exit_prices, sl_prices, rtol=1e-6)
        tp_hit = trade_mask & np.isclose(exit_prices, tp_prices, rtol=1e-6)
        close_hit = trade_mask & ~sl_hit & ~tp_hit

        total_trades = int(trade_mask.sum())
        win_rate = (len(wins) / max(total_trades, 1)) * 100.0
        loss_rate = (len(losses) / max(total_trades, 1)) * 100.0
        action_counts = np.array([(actions == i).sum() for i in range(len(SignalDirection))], dtype=np.float64)
        action_ratios = action_counts / max(action_counts.sum(), 1.0)

        # Scalar summaries mirroring the previous dashboard.
        self.tb_writer.add_scalar("dashboard/balance_last", float(balance[-1]) if len(balance) else 0.0, self.num_timesteps)
        self.tb_writer.add_scalar("dashboard/win_rate_pct", float(win_rate), self.num_timesteps)
        self.tb_writer.add_scalar("dashboard/loss_rate_pct", float(loss_rate), self.num_timesteps)
        self.tb_writer.add_scalar("dashboard/action_ratio_buy", float(action_ratios[SignalDirection.BUY]), self.num_timesteps)
        self.tb_writer.add_scalar("dashboard/action_ratio_sell", float(action_ratios[SignalDirection.SELL]), self.num_timesteps)
        self.tb_writer.add_scalar("dashboard/action_ratio_hold", float(action_ratios[SignalDirection.HOLD]), self.num_timesteps)
        self.tb_writer.add_scalar("dashboard/exit_count_sl", float(sl_hit.sum()), self.num_timesteps)
        self.tb_writer.add_scalar("dashboard/exit_count_tp", float(tp_hit.sum()), self.num_timesteps)
        self.tb_writer.add_scalar("dashboard/exit_count_close", float(close_hit.sum()), self.num_timesteps)

        if total_trades > 0:
            cum_pnl = np.cumsum(trade_pnl[trade_mask])
            self.tb_writer.add_scalar("dashboard/cumulative_trade_pnl", float(cum_pnl[-1]), self.num_timesteps)
            self.tb_writer.add_scalar("dashboard/trade_pnl_mean", float(np.mean(trade_pnl[trade_mask])), self.num_timesteps)

            # Histograms for trade quality analysis.
            self.tb_writer.add_histogram("dashboard/trade_pnl_hist", trade_pnl[trade_mask], self.num_timesteps)

            if len(wins) > 0:
                self.tb_writer.add_histogram("dashboard/wins_hist", wins, self.num_timesteps)

            if len(losses) > 0:
                self.tb_writer.add_histogram("dashboard/losses_hist", losses, self.num_timesteps)


class ModelTrainer:
    def __init__(
        self,
        features: list[str],
        initial_balance: float = 10_000,
        risk_per_trade: float = 0.01,
        fee_percent: float = 0.001,
        sl_multiplier: float = 1.0,
        tp_multiplier: float = 2.0,
        window_size: int = 1,
        learning_rate: float = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        ent_coef: float = 0.01,
        vec_env_type: str = "dummy",
        device: str = "auto",
        n_envs: int = 1,
        plot_every: int = 2048,
        model_save_path: str = "framework/ai/models",
        tensorboard_log: str = "framework/ai/training/tensorboard",
        tb_log_name: str = "PPO",
    ) -> None:
        self.features = features
        self.model_save_path = model_save_path
        self.initial_balance = initial_balance
        self.risk_per_trade = risk_per_trade
        self.fee_percent = fee_percent
        self.sl_multiplier = sl_multiplier
        self.tp_multiplier = tp_multiplier
        self.window_size = window_size

        # Model hyperparameters
        self.learning_rate = learning_rate
        self.n_steps = n_steps
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.ent_coef = ent_coef

        # Performance and runtime controls
        self.vec_env_type = vec_env_type.lower()
        self.device = device
        self.n_envs = n_envs
        self.plot_every = plot_every
        self.tensorboard_log = tensorboard_log
        self.tb_log_name = tb_log_name

    def train(self, train_df: pd.DataFrame) -> None:
        # 1. Create Vectorized Environment (Training on Train Set)
        env = self.__create_vectorized_environment(train_df)

        # 2. Calculate total_timesteps based on training data size and vectorization settings.
        total_timesteps = self.__calculate_timesteps(train_df)

        print(f"Training on {len(self.features)} features for {total_timesteps} steps.")

        # 3. Initialize On-Policy RL Model
        model = self.__create_model(env)

        # 4. Create TradeMetrics Callback
        callback_list = self.__create_callback_list()

        # 5. Train
        print(f"Training for {total_timesteps} steps...")
        try:
            model.learn(total_timesteps=total_timesteps, callback=callback_list, tb_log_name=self.tb_log_name)

            # 6. Save
            model.save(self.model_save_path)

            print(f"Training completed. Model saved to {self.model_save_path}.")
        except KeyboardInterrupt:
            print("Training interrupted manually. Skipping model save...")

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
            "win_rate_%": (len(wins) / len(trades) * 100) if len(trades) > 0 else 0.0,
            "avg_win_%": (sum(t.pnl_pct for t in wins) / len(wins) * 100) if len(wins) > 0 else 0.0,
            "avg_loss_%": (sum(t.pnl_pct for t in losses) / len(losses) * 100) if len(losses) > 0 else 0.0,
            "avg_trade_pnl_%": (sum(t.pnl_pct for t in trades) / len(trades) * 100) if len(trades) > 0 else 0.0,
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

    def __create_vectorized_environment(self, df: pd.DataFrame) -> VecEnv:
        if self.vec_env_type == "dummy" and self.n_envs == 1:
            return DummyVecEnv([lambda: self.__create_environment(df)])

        if self.vec_env_type == "subproc" and self.n_envs > 1:
            env_fns = [lambda d=df: self.__create_environment(d) for _ in range(self.n_envs)]
            return SubprocVecEnv(env_fns)

        raise ValueError(f"Unsupported vec_env_type: {self.vec_env_type}")

    def __calculate_timesteps(self, train_df: pd.DataFrame) -> int:
        num_candles = len(train_df) - self.window_size
        steps_per_env = num_candles * 1.2

        raw_total = steps_per_env * self.n_envs
        rolout_size = self.n_steps * self.n_envs

        total_timesteps = (raw_total // rolout_size) * rolout_size

        return total_timesteps

    def __create_model(self, environment: VecEnv) -> PPO:
        # policy_kwargs = dict(
        #     features_extractor_class=TCNFeatureExtractor,
        #     features_extractor_kwargs=dict(features_dim=64),
        #     net_arch=dict(pi=[32, 32], vf=[32, 32]),
        # )

        model = PPO(
            "MlpPolicy",
            environment,
            learning_rate=self.learning_rate,
            n_steps=self.n_steps,
            batch_size=self.batch_size,
            n_epochs=self.n_epochs,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            ent_coef=self.ent_coef,
            device=self.device,
            # policy_kwargs=policy_kwargs,
        )

        # Persist SB3 training metrics (including console table fields) to CSV and TensorBoard.
        log_dir = Path(self.tensorboard_log) / self.tb_log_name
        log_dir.mkdir(parents=True, exist_ok=True)
        model.set_logger(configure(str(log_dir), ["stdout", "csv", "tensorboard"]))

        return model

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

    def __create_callback_list(self) -> CallbackList:
        trade_metrics = TradeMetricsCallback(plot_every=self.plot_every)

        return CallbackList([trade_metrics])
