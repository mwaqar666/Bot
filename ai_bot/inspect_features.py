import sys
import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt

# Add parent path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from ai_bot.data_engine.feature_engineer import FeatureEngineer
from ai_bot.backtest import load_and_process_data


class FeatureInspector(FeatureEngineer):
    """
    Extends FeatureEngineer to verify distributions and visualize feature relationships.
    """

    def inspect_and_plot(self, csv_path: str):
        print(f"Loading {csv_path}...")

        # 0. Load Raw Data
        raw_df = pd.read_csv(csv_path)
        print(f"\n[Raw Data Check] NaNs present: {raw_df.isnull().sum().sum()}")
        if raw_df.isnull().sum().sum() > 0:
            print("  Columns with NaNs:")
            print(raw_df.isnull().sum()[raw_df.isnull().sum() > 0])

        # 1. Load and Process (Using shared loader)
        df, features = load_and_process_data(csv_path)

        print(f"\n[Processed Data Check] Final Rows: {len(df)}")
        print(f"NaNs after FE: {df.isnull().sum().sum()}")
        print(f"Features: {features}")

        # 2. Check Ranges
        self._print_stats(df, features)

        # 3. Visualization
        self._plot_price_chart(df)
        self._plot_feature_groups(df)

    def _plot_price_chart(self, df: pd.DataFrame):
        """Plots Price and Volume for the last 1000 candles."""
        print("Generating Price Chart...")
        data = df.iloc[-1000:]

        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=(15, 8), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
        )

        # Price
        x_axis = range(len(data))

        ax1.plot(x_axis, data["close"], label="Close Price", color="black")
        if "ema_fast" in data.columns:
            ax1.plot(
                x_axis, data["ema_fast"], label="EMA Fast", color="green", alpha=0.5
            )
        if "ema_slow" in data.columns:
            ax1.plot(x_axis, data["ema_slow"], label="EMA Slow", color="red", alpha=0.5)

        ax1.set_title("Price Action (Last 1000 Candles)")
        ax1.set_ylabel("Price")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Volume
        ax2.bar(x_axis, data["volume"], color="gray", alpha=0.5, label="Volume")
        ax2.set_ylabel("Volume")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig("price_chart.png")
        print("Saved price_chart.png")
        plt.close()

    def _print_stats(self, df: pd.DataFrame, features: list):
        print("\n--- Feature Statistics ---")
        stats = (
            df[features]
            .describe()
            .T[["min", "max", "mean", "std", "25%", "50%", "75%"]]
        )
        pd.set_option("display.max_rows", None)
        print(stats)

        # Alert on Outliers
        outliers = stats[(stats["min"] < -5) | (stats["max"] > 5)]
        if not outliers.empty:
            print("\n[WARNING] Potential Unnormalized Features (Mag > 5):")
            print(outliers.index.tolist())
        else:
            print("\n[OK] All features seem normalized (-5 to 5 range).")

    def _plot_feature_groups(self, df: pd.DataFrame):
        """Plot features grouped by category (Technical, Volume, Time, Normalized)."""
        plot_groups = {
            "Normalized_Inputs": [
                "log_ret_norm",
                "rsi_norm",
                "ema_spread",
                "macd_hist",
                "obv_slope",
                "volume_ratio_norm",
                "atr",
                "bb_width",
            ],
            "Price_Action_Features": [
                "candle_range",
                "candle_body",
                "upper_shadow",
                "lower_shadow",
            ],
            "Time_Features": ["sin_hour", "cos_hour", "sin_day", "cos_day"],
            "Higher_Timeframe_Context": [
                "trend_30m",
                "rsi_30m",
                "volatility_30m",
                "trend_1h",
                "rsi_1h",
                "volatility_1h",
                "trend_4h",
                "rsi_4h",
                "volatility_4h",
            ],
        }

        for title, cols in plot_groups.items():
            valid_cols = [c for c in cols if c in df.columns]
            if not valid_cols:
                continue

            n_cols = len(valid_cols)
            # Create subplots grid
            rows = (n_cols + 2) // 3  # 3 plots per row
            fig, axes = plt.subplots(rows, 3, figsize=(15, 3 * rows))
            fig.suptitle(f"{title} - Distribution & timeline", fontsize=16)
            axes = axes.flatten()

            for i, col in enumerate(valid_cols):
                ax = axes[i]

                # Plot Histogram (Distribution)
                data = df[col].iloc[-1000:]  # Last 1000 points
                ax.hist(
                    data, bins=50, alpha=0.7, color="blue", density=True, label="Dist"
                )

                # Overlay Line Plot on Twin Axis (Time Series)
                ax2 = ax.twinx()
                ax2.plot(
                    data.values, color="red", alpha=0.3, linewidth=1, label="Series"
                )

                ax.set_title(col)
                ax.grid(True, alpha=0.3)

            # Hide unused subplots
            for j in range(i + 1, len(axes)):
                axes[j].axis("off")

            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            filename = f"plot_{title}.png"
            plt.savefig(filename)
            print(f"Saved {filename}")
            plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect and Plot Features for AI Bot")
    parser.add_argument(
        "csv_path",
        nargs="?",
        default="ai_bot/data_engine/BTC_USDT_USDT_15m.csv",
        help="Path to input CSV",
    )

    args = parser.parse_args()

    target_csv = args.csv_path
    if not os.path.exists(target_csv):
        print(f"Error: File {target_csv} not found.")
        sys.exit(1)

    inspector = FeatureInspector()
    inspector.inspect_and_plot(target_csv)
