import pandas as pd
from datetime import datetime

import config

from framework.analysis.technical_indicators import TechnicalIndicators
from framework.data.data_loader import DataLoader
from framework.data.data_types import Position, TradeSignal, SignalDirection
from framework.inference import AI_Analyst
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
        self.ai_analyst = AI_Analyst()
        self.ai_analyst.load_model()
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

        try:
            df = self.__fetch_market_data()

            # Create TradeSignal from AI Action
            signal_direction = self.ai_analyst.analyze(df)

            print(f"AI: {signal_direction.upper()}")

            # Calculate SL/TP based on ATR
            current_price = df["close"].iloc[-1]
            atr = df["atr"].iloc[-1]

            sl_price = 0.0
            tp_price = 0.0

            if signal_direction == SignalDirection.BUY:
                sl_price = current_price - (atr * config.SL_ATR_MULTIPLIER)
                tp_price = current_price + (atr * config.TP_ATR_MULTIPLIER)
            elif signal_direction == SignalDirection.SELL:
                sl_price = current_price + (atr * config.SL_ATR_MULTIPLIER)
                tp_price = current_price - (atr * config.TP_ATR_MULTIPLIER)

            ai_signal = TradeSignal(direction=signal_direction, price=current_price, stop_loss=sl_price, take_profit=tp_price, reason=f"AI Model (Conf: {ai_confidence:.2f})", confidence=ai_confidence)

            position = self.executor.get_position(self.symbol)

            if position is not None:
                self.__handle_open_position(position, ai_signal)
            else:
                self.__check_for_entry(ai_signal)
        except ValueError as e:
            print(f"Error: Could not fetch data. Skipping cycle. {e}")
            return

    def __fetch_market_data(self) -> pd.DataFrame:
        """
        Fetches market data using DataLoader and adds technical indicators.

        Args:
            None

        Returns:
            pd.DataFrame: DataFrame with OHLCV data and technical indicators or ValueError if failed.
        """
        df = self.data_loader.fetch_historical_data(self.symbol, self.timeframe, days=config.DATA_LOOKBACK_DAYS)
        if df is None:
            raise ValueError("Failed to fetch market data.")

        return self.technical_indicators.add_indicators(df)

    def __handle_open_position(self, position: Position, signal: TradeSignal) -> None:
        """
        Manages exits and flips for open positions.

        Args:
            position (Position): Current open position object.
            signal (TradeSignal): Signal from the AI.

        Returns:
            None
        """
        self.__log_position_status(position)

        should_flip = self.__check_reversal(position, signal)
        if should_flip:
            self.__execute_flip(position, signal)

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

    def __check_reversal(self, position: Position, signal: TradeSignal) -> bool:
        """
        Checks if a reversal signal exists.

        Args:
            position (Position): Current open position.
            signal (TradeSignal): New strategy signal.

        Returns:
            bool: True if a reversal is indicated, False otherwise.
        """
        if position.side == "long" and signal.direction == SignalDirection.SELL:
            print("Exit Signal: Reversal detected (Long -> Short)")
            return True
        elif position.side == "short" and signal.direction == SignalDirection.BUY:
            print("Exit Signal: Reversal detected (Short -> Long)")
            return True
        return False

    def __execute_flip(self, position: Position, signal: TradeSignal) -> None:
        """
        Executes Stop & Reverse: Closes current position and attempts to open new one.

        Args:
            position (Position): The position to close.
            signal (TradeSignal): The signal indicating the new direction.

        Returns:
            None
        """
        print("Executing EXIT Logic...")
        print(f"Cancelling open orders and closing {position.side} position...")

        self.executor.cancel_orders(self.symbol)

        close_side = "sell" if position.side == "long" else "buy"
        # self.executor.place_order(self.symbol, close_side, position.amount)

        self.__attempt_flip_entry(close_side, position, signal)

    def __attempt_flip_entry(self, direction: str, position: Position, signal: TradeSignal) -> None:
        """
        Attempts to enter a new position after closing the old one.

        Args:
            direction (str): The direction of the closed trade (which becomes the new entry direction).
            position (Position): The closed position object (for PnL estimation).
            signal (TradeSignal): The valid trade signal used for calculating new entry.

        Returns:
            None
        """
        if signal.direction != direction:
            print(f"Position Closed. No Flip (AI is {signal.direction}). Going to Cash.")
            return

        print(f"Algo confirms {direction.upper()}. Executing FLIP Entry...")

        balance = self.executor.get_balance("USDT")
        estimated_balance = balance + float(position.unrealized_pnl)

        amount = self.__calculate_position_size(estimated_balance, signal)
        if amount > 0:
            self.__place_entry_order(direction, amount, signal)
            print("Flip Complete via Stop & Reverse!")

    def __check_for_entry(self, signal: TradeSignal) -> None:
        """
        Checks for entry signals.

        Args:
            signal (TradeSignal): AI signal.

        Returns:
            None
        """
        if signal.direction != SignalDirection.NONE:
            self.__execute_entry(signal.direction, signal)
        else:
            print("No valid signal. Waiting...")

    def __execute_entry(self, direction: str, signal: TradeSignal) -> None:
        """
        Calculates size and places fresh entry order.

        Args:
            direction (str): 'buy' or 'sell'.
            signal (TradeSignal): Validated trade signal.

        Returns:
            None
        """
        print(f"SIGNAL CONFIRMED: {direction.upper()} | Reason: {signal.reason}")
        print(f"Price: {signal.price} | SL: {signal.stop_loss:.2f} | TP: {signal.take_profit:.2f}")

        balance = self.executor.get_balance("USDT")
        if balance < 10:
            print("Balance too low to trade.")
            return

        amount = self.__calculate_position_size(balance, signal)
        if amount > 0:
            print(f"Placing {direction.upper()} order for {amount} {self.symbol}...")
            self.__place_entry_order(direction, amount, signal)
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
