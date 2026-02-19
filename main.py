import argparse
import time
import schedule
import config
from framework.trading.bot import TradingBot


def main():
    parser = argparse.ArgumentParser(description="Crypto Trading Bot Entry Point")

    # Optional overrides
    parser.add_argument("--symbol", type=str, default=config.SYMBOL, help=f"Override symbol (e.g. BTC/USDT). Default: {config.SYMBOL}")
    parser.add_argument("--timeframe", type=str, default=config.TIMEFRAME, help=f"Override timeframe (e.g. 5m, 1h). Default: {config.TIMEFRAME}")
    parser.add_argument("--days", type=int, default=config.DATA_LOOKBACK_DAYS, help=f"Number of days of data (only for 'data' mode). Default: {config.DATA_LOOKBACK_DAYS}")

    args = parser.parse_args()

    try:
        # Override config with args if provided
        if args.symbol:
            config.SYMBOL = args.symbol
        if args.timeframe:
            config.TIMEFRAME = args.timeframe

        print(f"Starting Bot for {config.SYMBOL} on {config.TIMEFRAME}...")

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
    main()
