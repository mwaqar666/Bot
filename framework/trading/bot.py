import pandas as pd
from datetime import datetime
from typing import Optional

import config

from framework.analysis.technical_indicators import TechnicalIndicators
from framework.data.data_loader import DataLoader
from framework.data.data_types import Position, TradeSignal
from framework.strategies.momentum_strategy import MomentumStrategy
from framework.trading.execution import TradeExecutor


class TradingBot:
    def __init__(self) -> None:
        """
        Initializes the Trading Bot using configuration.

        Args:
            None

        Returns:
            None
        """
        print("Initializing Trading Bot...")

        if not config.API_KEY or not config.API_SECRET:
            raise ValueError("API Keys not found in .env file.")

        self.executor = TradeExecutor(config.API_KEY, config.API_SECRET, testnet=config.IS_TESTNET)
        self.symbol = config.SYMBOL
        self.timeframe = config.TIMEFRAME

        self.data_loader = DataLoader()
        self.strategy = MomentumStrategy()
        self.technical_indicators = TechnicalIndicators()

        # Set Leverage
        self.executor.set_leverage(self.symbol, config.MAX_LEVERAGE)
        print("Bot Initialized Successfully.")

    def run_cycle(self) -> None:
        """
        Main execution cycle logic.
        Runs fetch, analyze, and execute loop.

        Args:
            None

        Returns:
            None
        """
        print(f"\n--- Bot Cycle Started at {datetime.now().strftime('%H:%M:%S')} ---")

        df = self.__fetch_market_data()
        if df is None or df.empty:
            print("Error: Could not fetch data. Skipping cycle.")
            return

        algo_signal = self.strategy.analyze(df)
        print(f"ALGO: {algo_signal.direction.upper()}")

        position = self.executor.get_position(self.symbol)
        if position:
            self.__handle_open_position(position, algo_signal)
        else:
            self.__check_for_entry(algo_signal)

    def __fetch_market_data(self) -> Optional[pd.DataFrame]:
        """
        Fetches market data using DataLoader.

        Args:
            None

        Returns:
            Optional[pd.DataFrame]: DataFrame with OHLCV data or None if failed.
        """
        df = self.data_loader.fetch_historical_data(self.symbol, self.timeframe, days=config.DATA_LOOKBACK_DAYS)
        if df is not None and not df.empty:
            # Process features
            df = self.technical_indicators.add_indicators(df)
        return df

    def __handle_open_position(self, position: Position, algo_signal: TradeSignal, ai_decision: str) -> None:
        """
        Manages exits and flips for open positions.

        Args:
            position (Position): Current open position object.
            algo_signal (TradeSignal): Signal from the strategy.
            ai_decision (str): Decision from the AI model (buy/sell/hold).

        Returns:
            None
        """
        self.__log_position_status(position)

        should_flip = self.__check_reversal(position, algo_signal, ai_decision)
        if should_flip:
            self.__execute_flip(position, algo_signal)

    def __log_position_status(self, position: Position) -> None:
        """
        Logs current position status to console.

        Args:
            position (Position): Current open position object.

        Returns:
            None
        """
        print(f"OPEN POSITION: {position.side.upper()} {position.amount} {self.symbol}")
        print(f"Unrealized PnL: {position.unrealized_pnl} USDT")

    def __check_reversal(self, position: Position, algo_signal: TradeSignal, ai_decision: str) -> bool:
        """
        Checks if a reversal signal exists.

        Args:
            position (Position): Current open position.
            algo_signal (TradeSignal): New strategy signal.
            ai_decision (str): New AI decision.

        Returns:
            bool: True if a reversal is indicated, False otherwise.
        """
        if position.side == "long":
            if algo_signal.direction == "sell" or ai_decision == "sell":
                print("Exit Signal: Reversal detected (Long -> Short)")
                return True
        elif position.side == "short":
            if algo_signal.direction == "buy" or ai_decision == "buy":
                print("Exit Signal: Reversal detected (Short -> Long)")
                return True
        return False

    def __execute_flip(self, position: Position, algo_signal: TradeSignal) -> None:
        """
        Executes Stop & Reverse: Closes current position and attempts to open new one.

        Args:
            position (Position): The position to close.
            algo_signal (TradeSignal): The signal indicating the new direction.

        Returns:
            None
        """
        print("Executing EXIT Logic...")
        print(f"Cancelling open orders and closing {position.side} position...")

        self.executor.cancel_orders(self.symbol)

        close_side = "sell" if position.side == "long" else "buy"
        self.executor.place_order(self.symbol, close_side, position.amount)

        self.__attempt_flip_entry(close_side, position, algo_signal)

    def __attempt_flip_entry(self, direction: str, position: Position, algo_signal: TradeSignal) -> None:
        """
        Attempts to enter a new position after closing the old one.

        Args:
            direction (str): The direction of the closed trade (which becomes the new entry direction).
            position (Position): The closed position object (for PnL estimation).
            algo_signal (TradeSignal): The valid trade signal used for calculating new entry.

        Returns:
            None
        """
        if algo_signal.direction != direction:
            print(f"Position Closed. No Flip (Algo is {algo_signal.direction}). Going to Cash.")
            return

        print(f"Algo confirms {direction.upper()}. Executing FLIP Entry...")

        balance = self.executor.get_balance("USDT")
        estimated_balance = balance + float(position.unrealized_pnl)

        amount = self.__calculate_position_size(estimated_balance, algo_signal)
        if amount > 0:
            self.__place_entry_order(direction, amount, algo_signal)
            print("Flip Complete via Stop & Reverse!")

    def __check_for_entry(self, algo_signal: TradeSignal, ai_decision: str) -> None:
        """
        Checks confluence and executes fresh entries.

        Args:
            algo_signal (TradeSignal): Strategy signal.
            ai_decision (str): AI decision.

        Returns:
            None
        """
        final_direction = self.__get_confluence_direction(algo_signal, ai_decision)

        if final_direction != "none":
            self.__execute_entry(final_direction, algo_signal)
        else:
            print("No valid confluence signal. Waiting...")

    def __get_confluence_direction(self, algo_signal: TradeSignal, ai_decision: str) -> str:
        """
        Determines direction based on Algo and AI agreement.

        Args:
            algo_signal (TradeSignal): Strategy signal.
            ai_decision (str): AI decision.

        Returns:
            str: 'buy', 'sell', or 'none'.
        """
        if algo_signal.direction == "buy" and ai_decision == "buy":
            print(">>> CONFLUENCE DETECTED: STRONG BUY (LONG) <<<")
            return "buy"
        elif algo_signal.direction == "sell" and ai_decision == "sell":
            print(">>> CONFLUENCE DETECTED: STRONG SELL (SHORT) <<<")
            return "sell"
        elif algo_signal.direction != "none" and ai_decision != algo_signal.direction:
            print(f"CONFLICT: Algo={algo_signal.direction.upper()} vs AI={ai_decision.upper()}.")
            return "none"
        return "none"

    def __execute_entry(self, direction: str, algo_signal: TradeSignal) -> None:
        """
        Calculates size and places fresh entry order.

        Args:
            direction (str): 'buy' or 'sell'.
            algo_signal (TradeSignal): Validated trade signal.

        Returns:
            None
        """
        print(f"SIGNAL CONFIRMED: {direction.upper()} | Reason: {algo_signal.reason}")
        print(f"Price: {algo_signal.price} | SL: {algo_signal.stop_loss:.2f} | TP: {algo_signal.take_profit:.2f}")

        balance = self.executor.get_balance("USDT")
        if balance < 10:
            print("Balance too low to trade.")
            return

        amount = self.__calculate_position_size(balance, algo_signal)
        if amount > 0:
            print(f"Placing {direction.upper()} order for {amount} {self.symbol}...")
            self.__place_entry_order(direction, amount, algo_signal)
            print("Entry Order Placed with SL/TP!")

    def __place_entry_order(self, direction: str, amount: float, signal: TradeSignal) -> None:
        """
        Places the actual order with SL/TP.

        Args:
            direction (str): 'buy' or 'sell'.
            amount (float): Quantity to trade.
            signal (TradeSignal): Signal object containing SL/TP prices.

        Returns:
            None
        """
        self.executor.place_order(
            self.symbol,
            direction,
            amount,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
        )

    def __calculate_position_size(self, balance: float, signal: TradeSignal) -> float:
        """
        Calculates position size based on Risk % and Stop Loss distance.

        Args:
            balance (float): Account balance to base risk on.
            signal (TradeSignal): Signal containing price and stop_loss.

        Returns:
            float: Calculated position size (quantity).
        """
        if signal.price == 0:
            return 0.0

        risk_amt = balance * config.RISK_PER_TRADE
        sl_percent = self.__get_sl_percent(signal)

        # Position Size in USDT = Risk / SL%
        pos_size_usdt = risk_amt / sl_percent

        pos_size_usdt = self.__apply_leverage_cap(pos_size_usdt, balance)

        amount = pos_size_usdt / signal.price
        return round(amount, 3)

    def __get_sl_percent(self, signal: TradeSignal) -> float:
        """
        Calculates Stop Loss percentage distance.

        Args:
            signal (TradeSignal): Signal containing price and stop_loss.

        Returns:
            float: Decimal percentage distance (e.g., 0.01 for 1%).
        """
        sl_dist = abs(signal.price - signal.stop_loss)
        sl_percent = sl_dist / signal.price
        return sl_percent if sl_percent > 0 else 0.01

    def __apply_leverage_cap(self, pos_size_usdt: float, balance: float) -> float:
        """
        Caps position size based on max leverage.

        Args:
            pos_size_usdt (float): Raw position size in USDT.
            balance (float): Account balance.

        Returns:
            float: Capped position size in USDT.
        """
        max_pos_size = balance * config.MAX_LEVERAGE
        if pos_size_usdt > max_pos_size:
            pos_size_usdt = max_pos_size
            print(f"WARNING: Position size capped by Max Leverage ({config.MAX_LEVERAGE}x).")

        effective_leverage = pos_size_usdt / balance
        print(f"Dynamic Position Sizing: {pos_size_usdt:.2f} USDT (Lev: {effective_leverage:.2f}x)")
        return pos_size_usdt
