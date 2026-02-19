import argparse
import time
import schedule
import matplotlib.pyplot as plt
import seaborn as sns
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
    symbol = args.symbol if args.symbol else config.SYMBOL
    timeframe = args.timeframe if args.timeframe else config.TIMEFRAME
    days = int(args.days) if args.days else config.DATA_LOOKBACK_DAYS

    print("--- Data Analysis Mode ---")
    print(f"Symbol: {symbol}")
    print(f"Timeframe: {timeframe}")
    print(f"Lookback: {days} days")

    print("1. Fetching data... (IN PROGRESS)")

    loader = DataLoader()
    df = loader.fetch_historical_data(symbol, timeframe, days)

    if df is None or df.empty:
        raise ValueError("Error: No data fetched.")

    print(f"Fetched {len(df)} rows of data. (COMPLETED)")

    print("2. Calculating Technical Indicators... (IN PROGRESS)")

    ti = TechnicalIndicators()
    df = ti.add_indicators(df)

    print("Calculated Technical Indicators. (COMPLETED)")

    print("3. Saving Raw Data and Stats to CSV... (IN PROGRESS)")

    loader.save_to_csv(df, symbol, timeframe, "_raw")
    loader.save_to_csv(df.describe(), symbol, timeframe, "_stats")

    print("Raw Data and Stats saved to CSV. (COMPLETED)")

    print("4. Applying Normalization... (IN PROGRESS)")

    df = ti.fit_scalers(df)

    print(f"Normalized Features: {list(df.columns)}. (COMPLETED)")

    print("5. Saving Normalized Data and Stats to CSV... (IN PROGRESS)")

    loader.save_to_csv(df, symbol, timeframe, "_normalized")
    loader.save_to_csv(df.describe(), symbol, timeframe, "_norm_stats")

    print("Normalized Data and Stats saved to CSV. (COMPLETED)")

    corr_df = df.select_dtypes(include=["float64", "int64"])

    print("6. Calculating Correlation Matrix on Normalized Features... (IN PROGRESS)")

    correlation_matrix_pearson = corr_df.corr(method="pearson")
    correlation_matrix_spearman = corr_df.corr(method="spearman")

    corr_file_p = f"framework/data/corr_pearson_{symbol.replace('/', '_')}_{timeframe}.csv"
    corr_file_s = f"framework/data/corr_spearman_{symbol.replace('/', '_')}_{timeframe}.csv"
    correlation_matrix_pearson.to_csv(corr_file_p)
    correlation_matrix_spearman.to_csv(corr_file_s)

    print(f"Correlation Matrix saved to: {corr_file_p} and {corr_file_s}. (COMPLETED)")

    print("7. Generating Correlation Heatmaps... (IN PROGRESS)")

    plt.figure(figsize=(24, 20))
    sns.heatmap(correlation_matrix_pearson, annot=False, cmap="coolwarm", vmin=-1, vmax=1, linewidths=0.5)
    plt.title(f"Normalized Pearson Correlation Matrix - {symbol} {timeframe}")
    heatmap_file_p = f"framework/data/heatmap_pearson_{symbol.replace('/', '_')}_{timeframe}.png"
    plt.savefig(heatmap_file_p)
    plt.close()

    plt.figure(figsize=(24, 20))
    sns.heatmap(correlation_matrix_spearman, annot=False, cmap="coolwarm", vmin=-1, vmax=1, linewidths=0.5)
    plt.title(f"Normalized Spearman Correlation Matrix - {symbol} {timeframe}")
    heatmap_file_s = f"framework/data/heatmap_spearman_{symbol.replace('/', '_')}_{timeframe}.png"
    plt.savefig(heatmap_file_s)
    plt.close()

    print(f"Correlation Heatmaps saved to: {heatmap_file_p} and {heatmap_file_s}. (COMPLETED)")


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
