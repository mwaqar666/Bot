import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from typing import Tuple, List, Dict
from datetime import datetime

from framework.data.data_types import Trade


class CryptoTradingEnv(gym.Env):
    """
    A custom Trading Environment for OpenAI Gym / Gymnasium.
    Simulates a crypto exchange with support for Long/Short positions and intra-candle stop-losses.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        features: List[str] = [],
        initial_balance: float = 1000.0,
        fee_percent: float = 0.001,
        sl_atr_multiplier: float = 2.0,
        tp_atr_multiplier: float = 4.0,
        window_size: int = 10,
    ):
        """
        Initialize the Trading Environment.

        Args:
            df (pd.DataFrame): The historical market data (OHLCV + Indicators).
            features (List[str]): List of column names to use as AI observations.
            initial_balance (float): Starting capital in USDT.
            fee_percent (float): Trading fee per transaction (e.g., 0.001 for 0.1%).
            sl_atr_multiplier (float): Multiplier for ATR to set Stop Loss distance.
            tp_atr_multiplier (float): Multiplier for ATR to set Take Profit distance.
            window_size (int): Number of past lookback steps to return as observation.
        """

        self.df = df
        self.initial_balance = initial_balance
        self.fee_percent = fee_percent
        self.sl_mul = sl_atr_multiplier
        self.tp_mul = tp_atr_multiplier
        self.window_size = window_size

        # Data pointers
        self.current_step = window_size
        self.max_steps = len(df) - 1

        # Actions: 0=Hold, 1=Buy, 2=Sell
        self.action_space = spaces.Discrete(3)

        # Features to look at in market data (OHLCV + Indicators)
        self.features = features

        # Define Observation Space (Normalized values)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.window_size, len(self.features)),
            dtype=np.float32,
        )

        # Agent State initialization
        self.balance = initial_balance
        self.net_worth = initial_balance
        self.position = 0.0  # Positive = Long, Negative = Short
        self.in_position = False
        self.entry_price = 0.0
        self.entry_step = 0

        # Stop Orders State
        self._reset_stops()

        # Performance Tracking
        self.trade_history: List[Trade] = []

    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, dict]:
        """
        Resets the environment to the initial state.
        """
        super().reset(seed=seed)

        # Agent State initialization
        self.balance = self.initial_balance
        self.net_worth = self.initial_balance
        self.position = 0.0
        self.in_position = False
        self.entry_price = 0.0
        self.entry_step = 0

        # Stop Orders State
        self._reset_stops()

        # Performance Tracking
        self.trade_history = []

        # Reset Step Pointer to Window Size
        self.current_step = self.window_size

        return self._next_observation(), {}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """
        Execute one time step within the environment.
        """
        self.current_step += 1

        # 1. Fetch Market Data for the current step
        market_data = self._get_current_market_data()

        # Snapshot previous net worth for reward calculation
        prev_net_worth = self.net_worth

        # 2. Intra-Candle Stop Logic
        stop_triggered = self._check_intra_candle_stops(market_data)

        # 3. Execute Action
        action_executions = 0
        if not stop_triggered:
            action_executions = self._execute_agent_action(action, market_data)

        # 4. Update Valuation
        self.net_worth = self._calculate_net_worth(market_data["close"])

        # 5. Calculate Reward
        reward = self._calculate_reward(prev_net_worth, action_executions)

        # 6. Check Termination
        terminated, truncated = self._check_termination_conditions()

        # Build Info Dict
        info = {
            "net_worth": self.net_worth,
            "price": market_data["close"],
            "pos": self.position,
        }

        return self._next_observation(), reward, terminated, truncated, info

    def render(self, mode="human"):
        """Renders the environment stats to the console."""
        if mode == "human":
            profit = self.net_worth - self.initial_balance
            print(f"Step: {self.current_step}")
            print(f"Balance: {self.balance:.2f}")
            print(f"Net Worth: {self.net_worth:.2f}")
            print(f"Profit: {profit:.2f}")
            print(f"Positions: {self.position}")

            # Simple metrics
            if len(self.trade_history) > 0:
                wins = [t for t in self.trade_history if t.pnl > 0]
                win_rate = len(wins) / len(self.trade_history) * 100
                print(f"Trades: {len(self.trade_history)}")
                print(f"Win Rate: {win_rate:.2f}%")

    # =========================================
    #           Helper Methods
    # =========================================

    def _next_observation(self) -> np.ndarray:
        """
        Extracts the WINDOW of feature vectors ending at current step.
        """
        start_index = self.current_step - self.window_size
        end_index = self.current_step
        obs = self.df.iloc[start_index:end_index][self.features].values
        return obs.astype(np.float32)

    def _get_current_market_data(self) -> Dict[str, float]:
        """Fetches necessary price/indicator data for the current step."""
        return self.df.iloc[self.current_step].to_dict()

    def _check_intra_candle_stops(self, data: Dict[str, float]) -> bool:
        """
        Checks if the Low or High of the current candle triggered a SL or TP.
        """
        if not self.in_position:
            return False

        current_low = data["low"]
        current_high = data["high"]
        triggered = False

        if self.position > 0:  # Long Position
            if current_low <= self.stop_loss_price:
                self._close_long(self.stop_loss_price, reason="SL")
                triggered = True
            elif current_high >= self.take_profit_price:
                self._close_long(self.take_profit_price, reason="TP")
                triggered = True

        elif self.position < 0:  # Short Position
            if current_high >= self.stop_loss_price:
                self._close_short(self.stop_loss_price, reason="SL")
                triggered = True
            elif current_low <= self.take_profit_price:
                self._close_short(self.take_profit_price, reason="TP")
                triggered = True

        return triggered

    def _execute_agent_action(self, action: int, data: Dict[str, float]) -> int:
        """
        Processes the AI's chosen action (Hold, Buy, Sell) and executes trades.
        """
        current_price = data["close"]
        current_atr = data["atr"]
        executions = 0

        # Action Map: 0=Hold, 1=Buy(Long), 2=Sell(Short)

        if not self.in_position:
            # Scenario: Flat -> Open Position
            if action == 1:
                self._open_long(current_price, current_atr)
                executions = 1
            elif action == 2:
                self._open_short(current_price, current_atr)
                executions = 1

        else:
            # Scenario: In Position -> Close/Flip
            if self.position > 0 and action == 2:
                # Long -> Short (Flip)
                self._close_long(current_price, reason="Signal")
                self._open_short(current_price, current_atr)
                executions = 2

            elif self.position < 0 and action == 1:
                # Short -> Long (Flip)
                self._close_short(current_price, reason="Signal")
                self._open_long(current_price, current_atr)
                executions = 2

        return executions

    def _calculate_reward(self, prev_net_worth: float, executions_count: int) -> float:
        """
        Calculates the Reward based on % Return and Trade Penalties.
        """
        epsilon = 1e-6

        if abs(prev_net_worth) < epsilon:
            profit_pct = 0.0
        else:
            profit_pct = (self.net_worth - prev_net_worth) / prev_net_worth

        return profit_pct * 100.0

    def _check_termination_conditions(self) -> Tuple[bool, bool]:
        """
        Checks if the episode should end.
        """
        terminated = False
        truncated = False

        if self.current_step >= self.max_steps:
            truncated = True

        if self.net_worth < self.initial_balance * 0.5:
            terminated = True

        return terminated, truncated

    def _calculate_net_worth(self, current_price: float) -> float:
        """Calculates total Equity (Cash + Unrealized PnL of open positions)."""
        if not self.in_position:
            return self.net_worth

        if self.position > 0:  # Long
            return self.position * current_price
        else:  # Short
            return abs(self.position) * (2 * self.entry_price - current_price)

    def _log_trade(self, pnl: float, exit_price: float, reason: str):
        """Records a closed trade to history"""
        # We need actual datetimes for the Trade object, but for RL simulations
        # we often just have steps. We'll use placeholders or fetch from DF if possible.
        # Here we just use current timestamp as placeholder for sim.
        now = datetime.now()

        side = "long" if self.position > 0 else "short"  # Note: position is closing here
        # Wait, self.position is cleared *after* this call in the close methods?
        # No, it's cleared after _log_trade call in the original code?
        # Let's check: _close_long calls _log_trade BEFORE clearing self.position. Correct.

        trade = Trade(
            symbol="SIMULATION",
            entry_price=self.entry_price,
            exit_price=exit_price,
            amount=abs(self.position),
            side=side,  # type: ignore
            pnl=pnl,
            entry_time=now,  # Placeholder, real RL would map step->time
            exit_time=now,
            reason=reason,
        )
        self.trade_history.append(trade)

    # =========================================
    #        Trade Execution Primitives
    # =========================================

    def _open_long(self, price: float, atr: float):
        equity = self.net_worth * (1 - self.fee_percent)
        self.position = equity / price
        self.entry_price = price
        self.entry_step = self.current_step
        self.in_position = True

        self.stop_loss_price = price - (atr * self.sl_mul)
        self.take_profit_price = price + (atr * self.tp_mul)

    def _close_long(self, price: float, reason: str = "Signal"):
        revenue = self.position * price * (1 - self.fee_percent)
        pnl = revenue - (self.position * self.entry_price)

        self.net_worth = revenue
        self._log_trade(pnl, price, reason)

        self.position = 0.0
        self.entry_price = 0.0
        self.in_position = False
        self._reset_stops()

    def _open_short(self, price: float, atr: float):
        equity = self.net_worth * (1 - self.fee_percent)
        self.position = -(equity / price)
        self.entry_price = price
        self.entry_step = self.current_step
        self.in_position = True

        self.stop_loss_price = price + (atr * self.sl_mul)
        self.take_profit_price = price - (atr * self.tp_mul)

    def _close_short(self, price: float, reason: str = "Signal"):
        raw_value = abs(self.position) * (2 * self.entry_price - price)
        self.net_worth = raw_value * (1 - self.fee_percent)

        pnl = self.net_worth - (abs(self.position) * self.entry_price)
        self._log_trade(pnl, price, reason)

        self.position = 0.0
        self.entry_price = 0.0
        self.in_position = False
        self._reset_stops()

    def _reset_stops(self):
        self.stop_loss_price = 0.0
        self.take_profit_price = 0.0
