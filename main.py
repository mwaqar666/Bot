import time
import schedule
import pandas as pd
from datetime import datetime
from typing import Dict, Tuple

import config
from execution import TradeExecutor
from indicators import calculate_technical_indicators
from strategy import check_for_signal, TradeSignal
from ai_bot.inference import AI_Analyst


class TradingBot:
    def __init__(self):
        print("Initializing Trading Bot...")

        # 1. Initialize API and Executor
        if not config.API_KEY or not config.API_SECRET:
            raise ValueError("API Keys not found in .env file.")

        self.executor = TradeExecutor(
            config.API_KEY, config.API_SECRET, testnet=config.IS_TESTNET
        )
        self.symbol = config.SYMBOL
        self.timeframe = config.TIMEFRAME

        # 2. Initialize AI
        print("Waking up AI Analyst...")
        self.ai_analyst = AI_Analyst()
        self.ai_analyst.load_model()

        # 3. Set Leverage
        self.executor.set_leverage(self.symbol, config.MAX_LEVERAGE)
        print("Bot Initialized Successfully.")

    def run_cycle(self):
        """Main execution cycle logic."""
        print(f"\n--- Bot Cycle Started at {datetime.now().strftime('%H:%M:%S')} ---")

        # 1. Fetch Data
        df, context_dfs = self._fetch_market_data()
        if df.empty:
            print("Error: Could not fetch data. Skipping cycle.")
            return

        # 2. Analyze
        df = calculate_technical_indicators(df)
        algo_signal, ai_decision, ai_conf = self._analyze_market(df, context_dfs)

        print(
            f"ALGO: {algo_signal.direction.upper()} | AI: {ai_decision.upper()} (Conf: {ai_conf:.2f})"
        )

        # 3. Execution
        position = self.executor.get_position(self.symbol)

        if position:
            self._handle_open_position(position, algo_signal, ai_decision)
        else:
            self._check_for_entry(algo_signal, ai_decision)

    def _fetch_market_data(self) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """Fetches base 15m data and additional context timeframes."""
        # 1. Fetch Base Data
        df = self.executor.fetch_ohlcv(self.symbol, self.timeframe)

        # 2. Fetch Context Data
        context_dfs = {}
        for tf in ["30m", "1h", "4h"]:
            try:
                ctx_df = self.executor.fetch_ohlcv(self.symbol, tf, limit=100)
                if not ctx_df.empty:
                    context_dfs[tf] = ctx_df
            except Exception as e:
                print(f"Warning: Could not fetch {tf}: {e}")

        return df, context_dfs

    def _analyze_market(
        self, df: pd.DataFrame, context_dfs: Dict
    ) -> Tuple[TradeSignal, str, float]:
        """Gets signals from Strategy (Algo) and AI."""
        # Algo Signal
        algo_signal = check_for_signal(df, config)

        # AI Signal
        ai_decision, ai_conf = self.ai_analyst.analyze(df, additional_dfs=context_dfs)

        return algo_signal, ai_decision, ai_conf

    def _handle_open_position(
        self, position: Dict, algo_signal: TradeSignal, ai_decision: str
    ):
        """Manages exits and flips for open positions."""
        print(
            f"OPEN POSITION: {position['side'].upper()} {position['amount']} {self.symbol}"
        )
        print(f"Unrealized PnL: {position['unrealized_pnl']} USDT")

        should_flip = False

        # Check Reversal Conditions
        if position["side"] == "long":
            if algo_signal.direction == "sell" or ai_decision == "sell":
                print("Exit Signal: Reversal detected (Long -> Short?)")
                should_flip = True
        elif position["side"] == "short":
            if algo_signal.direction == "buy" or ai_decision == "buy":
                print("Exit Signal: Reversal detected (Short -> Long?)")
                should_flip = True

        if should_flip:
            self._execute_flip(position, algo_signal)

    def _execute_flip(self, position: Dict, algo_signal: TradeSignal):
        """Executes Stop & Reverse (Close current, potentially open new)."""
        print("Executing EXIT Logic...")

        # 1. Close Existing
        print(f"Cancelling open orders and closing {position['side']} position...")
        self.executor.cancel_orders(self.symbol)

        current_side = position["side"]
        close_side = "sell" if current_side == "long" else "buy"

        self.executor.place_order(self.symbol, close_side, position["amount"])

        # 2. Open New? (Only if Algo confirms valid direction for SL/TP)
        # Identify intended new direction based on the close side (which is the new direction)
        # Logic: If we closed a Long (Sell), the flip direction is Sell.
        new_direction = close_side

        if algo_signal.direction == new_direction:
            print(f"Algo confirms {new_direction.upper()}. Executing FLIP Entry...")

            # Calculate size based on approximated new balance
            balance = self.executor.get_balance("USDT")
            unrealized = float(position.get("unrealized_pnl", 0))
            # Safety: If PnL updates slowly, this might be slightly off, but safe enough for MVP
            estimated_balance = balance + unrealized

            amount = self._calculate_position_size(estimated_balance, algo_signal)

            if amount > 0:
                print(
                    f"Placing FLIP Order with SL: {algo_signal.stop_loss} TP: {algo_signal.take_profit}"
                )
                self.executor.place_order(
                    self.symbol,
                    new_direction,
                    amount,
                    stop_loss=algo_signal.stop_loss,
                    take_profit=algo_signal.take_profit,
                )
                print("Flip Complete via Stop & Reverse!")
        else:
            print(
                f"Position Closed. No Flip (Algo is {algo_signal.direction}). Going to Cash."
            )

    def _check_for_entry(self, algo_signal: TradeSignal, ai_decision: str):
        """Checks confluence and executes fresh entries."""
        final_direction = "none"

        # Confluence Check
        if algo_signal.direction == "buy" and ai_decision == "buy":
            final_direction = "buy"
            print(">>> CONFLUENCE DETECTED: STRONG BUY (LONG) <<<")
        elif algo_signal.direction == "sell" and ai_decision == "sell":
            final_direction = "sell"
            print(">>> CONFLUENCE DETECTED: STRONG SELL (SHORT) <<<")
        elif algo_signal.direction != "none" and ai_decision != algo_signal.direction:
            print(
                f"CONFLICT: Algo={algo_signal.direction.upper()} vs AI={ai_decision.upper()}. NO TRADE."
            )

        if final_direction != "none":
            self._execute_entry(final_direction, algo_signal)
        else:
            print("No valid confluence signal. Waiting...")

    def _execute_entry(self, direction: str, algo_signal: TradeSignal):
        """Calculates size and places fresh entry order."""
        print(f"SIGNAL CONFIRMED: {direction.upper()} | Reason: {algo_signal.reason}")
        print(
            f"Price: {algo_signal.price} | SL: {algo_signal.stop_loss:.2f} | TP: {algo_signal.take_profit:.2f}"
        )

        balance = self.executor.get_balance("USDT")
        if balance < 10:  # Minimum balance check
            print("Balance too low to trade.")
            return

        amount = self._calculate_position_size(balance, algo_signal)

        if amount > 0:
            print(f"Placing {direction.upper()} order for {amount} {self.symbol}...")
            self.executor.place_order(
                self.symbol,
                direction,
                amount,
                stop_loss=algo_signal.stop_loss,
                take_profit=algo_signal.take_profit,
            )
            print("Entry Order Placed with SL/TP!")

    def _calculate_position_size(self, balance: float, signal: TradeSignal) -> float:
        """Calculates position size based on Risk % and Stop Loss distance."""
        risk_amt = balance * config.RISK_PER_TRADE

        # Distance to Stop Loss
        sl_dist = abs(signal.price - signal.stop_loss)

        if signal.price == 0:
            return 0.0

        sl_percent = sl_dist / signal.price

        if sl_percent == 0:
            sl_percent = 0.01  # Safety

        # Position Size in USDT = Risk / SL%
        # Example: Risk 10$, SL 1% dist -> Pos 1000$
        pos_size_usdt = risk_amt / sl_percent

        # Dynamic Leverage Check
        # If pos_size_usdt > balance * MAX_LEVERAGE, cap it.
        max_pos_size = balance * config.MAX_LEVERAGE

        if pos_size_usdt > max_pos_size:
            pos_size_usdt = max_pos_size
            print(
                f"WARNING: Position size capped by Max Leverage ({config.MAX_LEVERAGE}x)."
            )

        # Calculate Effective Leverage for display
        effective_leverage = pos_size_usdt / balance
        print(
            f"Dynamic Position Sizing: {pos_size_usdt:.2f} USDT (Effective Lev: {effective_leverage:.2f}x)"
        )

        # Convert to Quantity
        amount = pos_size_usdt / signal.price
        return round(amount, 3)


def start_bot():
    try:
        bot = TradingBot()

        # Schedule
        schedule.every(1).minutes.do(bot.run_cycle)

        # Run once immediately
        bot.run_cycle()

        print("Bot is running logic loop...")
        while True:
            schedule.run_pending()
            time.sleep(1)

    except Exception as e:
        print(f"Fatal Bot Error: {e}")


if __name__ == "__main__":
    start_bot()
