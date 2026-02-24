import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

from framework.data.data_types import Action, PositionSide


class TradingEnvironment(gym.Env):
    """
    A custom Trading Environment for OpenAI Gym / Gymnasium.
    This version implements a per-candle trading logic where positions
    are opened and closed within the same time step.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        window_size: int,
        features: list[str],
        initial_balance: float,
        risk_per_trade: float,
        fee_percent: float,
        slippage_percent: float,
        sl_multiplier: float,
        tp_multiplier: float,
        symbol: str,
    ) -> None:
        """
        Initialize the Trading Environment.

        Args:
            df (pd.DataFrame): The historical market data (OHLCV + Indicators).
            window_size (int): Number of past lookback steps to return as observation.
            features (List[str]): List of column names to use as AI observations.
            initial_balance (float): Starting capital in USDT.
            risk_per_trade (float): Percentage of balance to risk per trade (e.g., 0.01 for 1%).
            fee_percent (float): Trading fee per transaction (e.g., 0.001 for 0.1%).
            slippage_percent (float): Slippage per transaction (e.g., 0.001 for 0.1%).
            sl_multiplier (float): Multiplier for ATR to set Stop Loss distance.
            tp_multiplier (float): Multiplier for ATR to set Take Profit distance.
            symbol (str): The symbol of the asset to trade.
        """

        self.df = df
        self.window_size = window_size
        self.features = features
        self.initial_balance = initial_balance
        self.risk_per_trade = risk_per_trade
        self.fee_percent = fee_percent
        self.slippage_percent = slippage_percent
        self.sl_multiplier = sl_multiplier
        self.tp_multiplier = tp_multiplier
        self.symbol = symbol

        # Actions: 2=Buy, 0=Sell, 1=Hold
        self.action_space = spaces.Discrete(3)

        # Define Observation Space
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.window_size, len(self.features)),
            dtype=np.float32,
        )

        # Agent State initialization
        self.current_step = window_size
        self.max_steps = len(df)
        self.balance = initial_balance
        self.net_worth = initial_balance
        self.trade_history = []

    def reset(self, seed=None, options=None) -> tuple[np.ndarray, dict]:
        """
        Resets the environment to the initial state.
        """
        super().reset(seed=seed)

        self.current_step = self.window_size
        self.balance = self.initial_balance
        self.net_worth = self.initial_balance
        self.trade_history = []

        return self.__next_observation(), {}

    def step(self, action: Action) -> tuple[np.ndarray, float, bool, bool, dict]:
        """
        Execute one time step within the environment.
        In this per-candle model:
        1. Observation is data up to current_step.
        2. Trade is opened at candle[open], closed within same candle.
        3. Step transitions to current_step + 1.
        """
        # Current candle data
        candle = self.df.iloc[self.current_step]
        open_price = candle["open"]
        high_price = candle["high"]
        low_price = candle["low"]
        close_price = candle["close"]
        atr = candle["atr"]

        trade_pnl_pct = 0.0

        if action != Action.HOLD:
            # 1. Initialize Trade Parameters
            if action == Action.BUY:
                side = PositionSide.LONG
                entry_price = open_price * (1 + self.slippage_percent)
                sl_price = entry_price - (atr * self.sl_multiplier)
                tp_price = entry_price + (atr * self.tp_multiplier)
            else:  # Action.SELL
                side = PositionSide.SHORT
                entry_price = open_price * (1 - self.slippage_percent)
                sl_price = entry_price + (atr * self.sl_multiplier)
                tp_price = entry_price - (atr * self.tp_multiplier)

            # 2. Intra-candle SL/TP check
            final_exit_price = close_price

            if side == PositionSide.LONG:
                # Check Stop Loss first (conservative)
                if low_price <= sl_price:
                    final_exit_price = sl_price
                elif high_price >= tp_price:
                    final_exit_price = tp_price

                # Apply slippage on exit
                exit_price_with_slippage = final_exit_price * (1 - self.slippage_percent)

                # Profit/Loss Calculation
                trade_pnl_pct = (exit_price_with_slippage - entry_price) / entry_price

            else:  # SHORT
                if high_price >= sl_price:
                    final_exit_price = sl_price
                elif low_price <= tp_price:
                    final_exit_price = tp_price

                # Apply slippage on exit
                exit_price_with_slippage = final_exit_price * (1 + self.slippage_percent)

                # Profit/Loss Calculation (Entry - Exit)
                trade_pnl_pct = (entry_price - exit_price_with_slippage) / entry_price

            # Apply Trading Fee
            trade_pnl_pct -= self.fee_percent * 2  # Entry + Exit fee

            # Update Balance (Simulating risking a portion of balance)
            # For simplicity, we assume we trade with 'initial_balance * risk_per_trade'
            # and the PnL applies to that portion.
            trade_amount = self.balance * self.risk_per_trade
            profit_loss = trade_amount * trade_pnl_pct
            self.balance += profit_loss
            self.net_worth = self.balance

            self.trade_history.append(trade_pnl_pct)

        # 3. Advance to next step
        self.current_step += 1

        # 4. Check Termination
        terminated, truncated = self.__check_termination_conditions()

        # Clamp step to prevent IndexError on observation after truncation
        if truncated:
            self.current_step = len(self.df) - 1

        # 5. Get next observation
        obs = self.__next_observation()

        # 6. Reward: Logarithmic Return
        if action == Action.HOLD:
            reward = -0.001
        else:
            clamped_pnl = max(trade_pnl_pct, -0.9999)
            reward = float(np.log(1 + clamped_pnl))

        info = {
            "step": self.current_step,
            "net_worth": self.net_worth,
            "balance": self.balance,
            "trade_pnl": trade_pnl_pct,
            "reward": reward,
            "action": action,
        }

        return obs, float(reward), terminated, truncated, info

    def render(self, mode="human") -> None:
        """Renders the environment stats to the console."""
        if mode == "human":
            print(f"Step: {self.current_step} | Net Worth: {self.net_worth:.2f} | Balance: {self.balance:.2f}")

    # =========================================
    #           Helper Methods
    # =========================================

    def __next_observation(self) -> np.ndarray:
        end_index = self.current_step
        start_index = end_index - self.window_size
        return self.df.iloc[start_index:end_index][self.features].values.astype(np.float32)

    def __check_termination_conditions(self) -> tuple[bool, bool]:
        terminated = False
        truncated = False

        if self.current_step >= self.max_steps:
            truncated = True

        if self.net_worth < self.initial_balance * 0.1:
            terminated = True

        return terminated, truncated
