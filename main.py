import argparse
import time
import schedule
import config
from framework.trading.bot import TradingBot
from framework.data.data_loader import DataLoader
from framework.analysis.technical_indicators import TechnicalIndicators


def start_bot(args) -> None:
    """
    Entry point for the live trading bot.
    """
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


def download_and_analyze(args) -> None:
    """
    Entry point for downloading data and running analysis/indicators.
    Does NOT place trades.
    """
    try:
        symbol = args.symbol if args.symbol else config.SYMBOL
        timeframe = args.timeframe if args.timeframe else config.TIMEFRAME
        days = int(args.days) if args.days else config.DATA_LOOKBACK_DAYS

        print("--- Data Analysis Mode ---")
        print(f"Symbol: {symbol}")
        print(f"Timeframe: {timeframe}")
        print(f"Lookback: {days} days")

        # 1. Fetch Data
        loader = DataLoader()
        df = loader.fetch_historical_data(symbol, timeframe, days)

        if df is None or df.empty:
            print("Error: No data fetched.")
            return

        print(f"Fetched {len(df)} rows of data.")

        # 2. Add Indicators
        print("Calculating Technical Indicators...")
        ti = TechnicalIndicators()
        df = ti.add_indicators(df)

        print(f"Indicators added. Total columns: {len(df.columns)}")
        print(f"Columns: {df.columns.tolist()}")

        # 3. Save to CSV
        loader.save_to_csv(df, symbol, timeframe)

        # Optional: Print last 5 rows
        print("\nLast 5 rows:")
        print(df.tail())

    except Exception as e:
        print(f"Analysis Error: {e}")


def main():
    parser = argparse.ArgumentParser(description="Crypto Trading Bot Entry Point")

    # Mode selection
    parser.add_argument("mode", choices=["trade", "data"], default="data", help=f"Mode to run the bot in: 'trade' for live bot, 'data' for downloading/analyzing data. Default: {config.MODE}")

    # Optional overrides
    parser.add_argument("--symbol", type=str, default=config.SYMBOL, help=f"Override symbol (e.g. BTC/USDT). Default: {config.SYMBOL}")
    parser.add_argument("--timeframe", type=str, default=config.TIMEFRAME, help=f"Override timeframe (e.g. 5m, 1h). Default: {config.TIMEFRAME}")
    parser.add_argument("--days", type=int, default=config.DATA_LOOKBACK_DAYS, help=f"Number of days of data (only for 'data' mode). Default: {config.DATA_LOOKBACK_DAYS}")

    args = parser.parse_args()

    if args.mode == "trade":
        start_bot(args)
    elif args.mode == "data":
        download_and_analyze(args)


if __name__ == "__main__":
    main()
