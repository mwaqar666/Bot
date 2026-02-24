import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback

from framework.ai.training.trading_env import TradingEnvironment
from framework.ai.models.tcn_feature_extractor import TCNFeatureExtractor

# --- Configuration ---
MODEL_PATH = "framework/ai/models/ppo_tcn_v1"
LOG_DIR = "framework/ai/logs"

SYMBOL = "BTC/USDT"
INITIAL_BALANCE = 1000.0
RISK_PER_TRADE = 0.01  # Risk 1% of balance per trade
FEE_PERCENT = 0.001  # 0.1% maker/taker fee
SLIPPAGE_PERCENT = 0.0005  # 0.05% slippage
SL_MULTIPLIER = 2.0  # Stop Loss: 2x ATR
TP_MULTIPLIER = 4.0  # Take Profit: 4x ATR (2:1 Risk/Reward)
WINDOW_SIZE = 60  # Look back 60 candles (5 hours)
TOTAL_TIMESTEPS = 1_000_000


class TensorboardCallback(BaseCallback):
    """
    Custom callback for logging metrics to TensorBoard.
    Uses _on_rollout_end to log AVERAGES across the full 2048-step window,
    not just the last step (which is misleading when model HOLDs at the end).
    """

    def __init__(self, verbose=0):
        super(TensorboardCallback, self).__init__(verbose)
        self._rewards = []
        self._pnls = []

    def _on_step(self) -> bool:
        if self.locals.get("infos"):
            info = self.locals["infos"][0]
            self._rewards.append(info.get("reward", 0.0))
            self._pnls.append(info.get("trade_pnl", 0.0))

            # Balance is cumulative — last value is correct
            self.logger.record("custom/balance", info.get("balance", 0.0))
            self.logger.record("custom/net_worth", info.get("net_worth", 0.0))
        return True

    def _on_rollout_end(self) -> None:
        """Called once per 2048-step rollout. Log aggregates here."""
        if not self._rewards:
            return

        n_trades = sum(1 for p in self._pnls if p != 0.0)
        n_holds = sum(1 for p in self._pnls if p == 0.0)

        self.logger.record("custom/avg_reward", sum(self._rewards) / len(self._rewards))
        self.logger.record("custom/avg_trade_pnl", sum(self._pnls) / len(self._pnls))
        self.logger.record("custom/n_trades", n_trades)
        self.logger.record("custom/hold_rate_%", n_holds / len(self._pnls) * 100)

        self._rewards.clear()
        self._pnls.clear()


def train(train_df: pd.DataFrame, features: list[str]) -> None:
    print(f"Training on {len(features)} Features: {features}")

    # 1. Create Environment (Training on Train Set)
    env = DummyVecEnv(
        [
            lambda: TradingEnvironment(
                df=train_df,
                window_size=WINDOW_SIZE,
                features=features,
                initial_balance=INITIAL_BALANCE,
                risk_per_trade=RISK_PER_TRADE,
                fee_percent=FEE_PERCENT,
                slippage_percent=SLIPPAGE_PERCENT,
                sl_multiplier=SL_MULTIPLIER,
                tp_multiplier=TP_MULTIPLIER,
                symbol=SYMBOL,
            )
        ]
    )

    # 2. Initialize PPO
    print("Initializing PPO Agent...")

    policy_kwargs = dict(
        features_extractor_class=TCNFeatureExtractor,
        features_extractor_kwargs=dict(features_dim=128),
        net_arch=dict(pi=[64, 64], vf=[64, 64]),
    )

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=1e-4,  # Reduced from 3e-4 to stabilize high approx_kl
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,  # Discount factor — rewards agent for long-term profit
        ent_coef=0.01,  # Entropy bonus — mathematically prevents HOLD collapse
        verbose=1,
        policy_kwargs=policy_kwargs,
        tensorboard_log=LOG_DIR,
    )

    # 3. Train
    print(f"Training for {TOTAL_TIMESTEPS} steps...")
    try:
        model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=TensorboardCallback())
    except KeyboardInterrupt:
        print("Training interrupted manually. Saving current model...")

    # 4. Save
    print(f"Saving model to {MODEL_PATH}...")
    model.save(MODEL_PATH)
    print("Training complete.")


def evaluate(df: pd.DataFrame, features: list[str], label: str = "Validation") -> dict:
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

    env = TradingEnvironment(
        df=df,
        window_size=WINDOW_SIZE,
        features=features,
        initial_balance=INITIAL_BALANCE,
        risk_per_trade=RISK_PER_TRADE,
        fee_percent=FEE_PERCENT,
        slippage_percent=SLIPPAGE_PERCENT,
        sl_multiplier=SL_MULTIPLIER,
        tp_multiplier=TP_MULTIPLIER,
        symbol=SYMBOL,
    )

    model = PPO.load(MODEL_PATH, env=env)
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
        "total_return_%": (env.balance - INITIAL_BALANCE) / INITIAL_BALANCE * 100,
        "total_trades": len(trades),
        "win_rate_%": len(wins) / len(trades) * 100 if trades else 0.0,
        "avg_win_%": sum(wins) / len(wins) * 100 if wins else 0.0,
        "avg_loss_%": sum(losses) / len(losses) * 100 if losses else 0.0,
        "avg_trade_pnl_%": sum(trades) / len(trades) * 100 if trades else 0.0,
    }

    print(f"  Final Balance : ${metrics['final_balance']:.2f}  (Started: ${INITIAL_BALANCE:.2f})")
    print(f"  Total Return  : {metrics['total_return_%']:+.2f}%")
    print(f"  Total Trades  : {metrics['total_trades']}")
    print(f"  Win Rate      : {metrics['win_rate_%']:.1f}%")
    print(f"  Avg Win       : {metrics['avg_win_%']:+.3f}%")
    print(f"  Avg Loss      : {metrics['avg_loss_%']:+.3f}%")
    print(f"  Avg Trade PnL : {metrics['avg_trade_pnl_%']:+.3f}%")
    print("-----------------------------------")

    return metrics
