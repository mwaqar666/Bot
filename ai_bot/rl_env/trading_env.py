import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from typing import Tuple, List, Dict


class CryptoTradingEnv(gym.Env):
    """
    A custom Trading Environment for OpenAI Gym / Gymnasium.
    Simulates a crypto exchange with support for Long/Short positions and intra-candle stop-losses.

    Key Features:
    - Action Space: Discrete(3) -> Hold, Long, Short given as 0, 1, 2.
    - Observation Space: Continuous vector of technical indicators.
    - Intra-Candle Simulation: Checks Low/High of the *current* candle against SL/TP levels.
    - Fee Simulation: Deducts a percentage fee on every trade open/close.
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
        # Start at window_size to allow lookback (0 to window_size-1)
        self.current_step = window_size
        self.max_steps = len(df) - 1

        # Actions: 0=Hold, 1=Buy, 2=Sell
        self.action_space = spaces.Discrete(3)

        # Features to look at in market data (OHLCV + Indicators)
        self.features = features

        # Define Observation Space (Normalized values)
        # Shape: (Window_Size, Num_Features)
        # This allows the Transformer to see a sequence of history.
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
        self.trade_history = []

    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, dict]:
        """
        Resets the environment to the initial state.
        Called at the start of every episode (training loop).
        """
        super().reset(seed=seed)

        # Agent State initialization
        self.balance = self.initial_balance
        self.net_worth = self.initial_balance
        self.position = 0.0  # Positive = Long, Negative = Short
        self.in_position = False
        self.entry_price = 0.0
        self.entry_step = 0

        # Stop Orders State
        self._reset_stops()

        # Performance Tracking
        self.trade_history = []

        # Reset Step Pointer to Window Size (so we can look back immediately)
        self.current_step = self.window_size

        return self._next_observation(), {}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """
        Execute one time step within the environment.

        Logic Flow:
        1. Advance Time (Index).
        2. Check if existing positions hit SL/TP during the candle (Intra-candle check).
        3. Execute Agent's Action (Buy/Sell/Hold) if not stopped out.
        4. Calculate Rewards based on Net Worth change.
        5. Check for Episode Termination (End of Data or Bankruptcy).
        """
        self.current_step += 1

        # 1. Fetch Market Data for the current step
        market_data = self._get_current_market_data()

        # Snapshot previous net worth for reward calculation
        prev_net_worth = self.net_worth

        # 2. Intra-Candle Stop Logic
        # Returns True if a stop was triggered, meaning the position is now closed.
        stop_triggered = self._check_intra_candle_stops(market_data)

        # 3. Execute Action
        # Only allow new actions if we weren't just stopped out in this same candle.
        # (Simplifies logic: if stopped out, you wait for next bar to re-enter)
        action_executions = 0  # Count total executions (open + close) for fee penalty
        if not stop_triggered:
            action_executions = self._execute_agent_action(action, market_data)

        # 4. Update Valuation
        # Recalculate Net Worth based on new price (Mark-to-Market)
        self.net_worth = self._calculate_net_worth(market_data["close"])

        # 5. Calculate Reward
        # We assume 1 'execution' cost is akin to Paying the Fee.
        # The Fee is implicitly in Net Worth, but we add an extra 'Decision Penalty' to discourage overtrading.
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
                wins = [t for t in self.trade_history if t["pnl"] > 0]
                win_rate = len(wins) / len(self.trade_history) * 100
                print(f"Trades: {len(self.trade_history)}")
                print(f"Win Rate: {win_rate:.2f}%")

    # =========================================
    #           Helper Methods
    # =========================================

    def _next_observation(self) -> np.ndarray:
        """
        Extracts the WINDOW of feature vectors ending at current step.
        Returns shape: (window_size, num_features)
        """
        # Slice from (step - window_size) up to (step)
        start_index = self.current_step - self.window_size
        end_index = self.current_step

        # Note: We use values directly.
        # Ensure df is sorted by time (it usually is).
        obs = self.df.iloc[start_index:end_index][self.features].values

        return obs.astype(np.float32)

    def _get_current_market_data(self) -> Dict[str, float]:
        """Fetches necessary price/indicator data for the current step."""
        return self.df.iloc[self.current_step].to_dict()

    def _check_intra_candle_stops(self, data: Dict[str, float]) -> bool:
        """
        Checks if the Low or High of the current candle triggered a SL or TP.

        Returns:
            bool: True if a position was closed (stopped out), False otherwise.
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

        Returns:
            int: Number of actions (Open/Close) executed.
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
                executions = 2  # Close + Open = 2 Executions

            elif self.position < 0 and action == 1:
                # Short -> Long (Flip)
                self._close_short(current_price, reason="Signal")
                self._open_long(current_price, current_atr)
                executions = 2  # Close + Open = 2 Executions

        return executions

    def _calculate_reward(self, prev_net_worth: float, executions_count: int) -> float:
        """
        Calculates the Reward based on % Return and Trade Penalties.

        Reward = (Current_Net_Worth - Prev_Net_Worth) / Prev_Net_Worth * 100
        Penalty = -0.05 * Number_of_Executions (Transaction Cost Penalty)
        """
        epsilon = 1e-6  # Avoid division by zero

        if abs(prev_net_worth) < epsilon:
            profit_pct = 0.0
        else:
            profit_pct = (self.net_worth - prev_net_worth) / prev_net_worth

        # Reward is simply the Percentage Change in Net Worth
        # Fees are already deducted from Net Worth, so we don't need double penalty.
        # We multiply by 100 to make the numbers meaningful for the neural net (e.g. 0.01 -> 1.0)
        reward = profit_pct * 100.0

        return reward

    def _check_termination_conditions(self) -> Tuple[bool, bool]:
        """
        Checks if the episode should end.

        Returns:
            (terminated, truncated)
            terminated: True if "Game Over" (Bankruptcy).
            truncated: True if "Time Limit Reached" (End of Data).
        """
        terminated = False
        truncated = False

        # End of Dataset
        if self.current_step >= self.max_steps:
            truncated = True

        # Bankruptcy (50% Loss)
        if self.net_worth < self.initial_balance * 0.5:
            terminated = True

        return terminated, truncated

    def _calculate_net_worth(self, current_price: float) -> float:
        """Calculates total Equity (Cash + Unrealized PnL of open positions)."""
        if not self.in_position:
            return self.net_worth

        # Mark-to-Market Valuation
        if self.position > 0:  # Long
            return self.position * current_price
        else:  # Short
            # Short Value = Initial_Short_Value + (Entry - Current) * Size
            # Derived as: Position_Size * (2*Entry - Current)
            return abs(self.position) * (2 * self.entry_price - current_price)

    def _log_trade(self, pnl: float, exit_price: float, reason: str):
        """Records a closed trade to history"""
        self.trade_history.append(
            {
                "step": self.current_step,
                "entry_price": self.entry_price,
                "exit_price": exit_price,
                "pnl": pnl,
                "reason": reason,
                "duration": self.current_step - self.entry_step,
            }
        )

    # =========================================
    #        Trade Execution Primitives
    # =========================================

    def _open_long(self, price: float, atr: float):
        """
        Opens a Long position.
        1. Calculates Equity (minus fee).
        2. Sets Position Size.
        3. Sets SL/TP based on ATR.
        """
        equity = self.net_worth * (1 - self.fee_percent)
        self.position = equity / price
        self.entry_price = price
        self.entry_step = self.current_step
        self.in_position = True

        # Set Dynamic Stops
        self.stop_loss_price = price - (atr * self.sl_mul)
        self.take_profit_price = price + (atr * self.tp_mul)

    def _close_long(self, price: float, reason: str = "Signal"):
        """
        Closes a Long position.
        1. Calculates Revenue (minus fee).
        2. Updates Net Worth.
        3. Resets State.
        """
        revenue = self.position * price * (1 - self.fee_percent)
        pnl = revenue - (self.position * self.entry_price)  # Approximate PnL for stats

        self.net_worth = revenue
        self._log_trade(pnl, price, reason)

        self.position = 0.0
        self.entry_price = 0.0
        self.in_position = False
        self._reset_stops()

    def _open_short(self, price: float, atr: float):
        """
        Opens a Short position.
        1. Calculates Equity (minus fee).
        2. Sets Negative Position Size.
        3. Sets SL/TP based on ATR (Inverse direction).
        """
        equity = self.net_worth * (1 - self.fee_percent)
        self.position = -(equity / price)
        self.entry_price = price
        self.entry_step = self.current_step
        self.in_position = True

        # Set Dynamic Stops (Short: SL is Above, TP is Below)
        self.stop_loss_price = price + (atr * self.sl_mul)
        self.take_profit_price = price - (atr * self.tp_mul)

    def _close_short(self, price: float, reason: str = "Signal"):
        """
        Closes a Short position.
        1. Calculates Value based on Short PnL equation.
        2. Updates Net Worth.
        3. Resets State.
        """
        # Short PnL formula: Invested + (Entry - Current) * #Coins
        raw_value = abs(self.position) * (2 * self.entry_price - price)
        self.net_worth = raw_value * (1 - self.fee_percent)

        pnl = self.net_worth - (abs(self.position) * self.entry_price)  # Approx PnL
        self._log_trade(pnl, price, reason)

        self.position = 0.0
        self.entry_price = 0.0
        self.in_position = False
        self._reset_stops()

    def _reset_stops(self):
        """Resets stop prices to zero."""
        self.stop_loss_price = 0.0
        self.take_profit_price = 0.0
