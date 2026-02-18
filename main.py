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

    # 1. Fetch Data
    loader = DataLoader()
    df = loader.fetch_historical_data(symbol, timeframe, days)

    if df is None or df.empty:
        print("Error: No data fetched.")
        return

    # Ensure all numeric columns are float to avoid pandas/pandas-ta type warnings
    for col in ["open", "high", "low", "close", "volume"]:
        if col in df.columns:
            df[col] = df[col].astype(float)

    print(f"Fetched {len(df)} rows of data.")

    # 2. Add Indicators
    print("Calculating Technical Indicators...")

    ti = TechnicalIndicators()
    df = ti.add_indicators(df)

    # 3. Save Raw Output to CSV (for reference)
    loader.save_to_csv(df, symbol, timeframe)
    print("Raw Data saved to CSV.")

    # --- NORMALIZATION & CLEANING ---
    print("\nApplying In-Class Normalization...")

    # Call the new centralized normalization method
    # This relies on each indicator class implementing its own valid normalization logic
    df = ti.normalize_indicators(df)

    print(f"Normalized Features: {list(df.columns)}")

    # We perform analysis on the NORMALIZED data
    numeric_df = df.select_dtypes(include=["float64", "int64"])

    # 4. Correlation Analysis
    print("\nCalculating Correlation Matrix on Normalized Features...")

    if numeric_df.empty:
        print("Error: No numeric features left after normalization!")
        return

    correlation_matrix_pearson = numeric_df.corr(method="pearson")
    correlation_matrix_spearman = numeric_df.corr(method="spearman")

    # Save Correlation Matrices
    corr_file_p = f"framework/data/correlation_pearson_{symbol.replace('/', '_')}_{timeframe}.csv"
    corr_file_s = f"framework/data/correlation_spearman_{symbol.replace('/', '_')}_{timeframe}.csv"
    correlation_matrix_pearson.to_csv(corr_file_p)
    correlation_matrix_spearman.to_csv(corr_file_s)
    print(f"Pearson Correlation Matrix saved to: {corr_file_p}")
    print(f"Spearman Correlation Matrix saved to: {corr_file_s}")

    # 5. Generate Heatmap (Pearson)
    print("\nGenerating Normalized Correlation Heatmap (Pearson)...")
    plt.figure(figsize=(24, 20))
    sns.heatmap(correlation_matrix_pearson, annot=False, cmap="coolwarm", vmin=-1, vmax=1, linewidths=0.5)
    plt.title(f"Normalized Pearson Correlation Matrix - {symbol} {timeframe}")
    heatmap_file_p = f"framework/data/heatmap_pearson_norm_{symbol.replace('/', '_')}_{timeframe}.png"
    plt.savefig(heatmap_file_p)
    print(f"Heatmap saved to: {heatmap_file_p}")
    plt.close()

    # 6. Generate Heatmap (Spearman)
    print("\nGenerating Normalized Correlation Heatmap (Spearman)...")
    plt.figure(figsize=(24, 20))
    sns.heatmap(correlation_matrix_spearman, annot=False, cmap="coolwarm", vmin=-1, vmax=1, linewidths=0.5)
    plt.title(f"Normalized Spearman Correlation Matrix - {symbol} {timeframe}")
    heatmap_file_s = f"framework/data/heatmap_spearman_norm_{symbol.replace('/', '_')}_{timeframe}.png"
    plt.savefig(heatmap_file_s)
    print(f"Heatmap saved to: {heatmap_file_s}")
    plt.close()

    # Save Stats to CSV
    stats = df.describe()
    stats_file = f"framework/data/stats_norm_{symbol.replace('/', '_')}_{timeframe}.csv"
    stats.to_csv(stats_file)
    print(f"Stats saved to: {stats_file}")


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
