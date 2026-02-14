import sys
import pandas as pd
import numpy as np

try:
    from framework.analysis.technical_indicators import TechnicalIndicators
except ImportError as e:
    print(f"ImportError: {e}")
    sys.exit(1)


def test_indicators():
    print("Testing TechnicalIndicators...")

    # Create dummy OHLCV data
    dates = pd.date_range(start="2023-01-01", periods=100, freq="15min")
    df = pd.DataFrame({"open": np.random.rand(100) * 100, "high": np.random.rand(100) * 105, "low": np.random.rand(100) * 95, "close": np.random.rand(100) * 100, "volume": np.random.rand(100) * 1000}, index=dates)

    ti = TechnicalIndicators()

    try:
        df_processed = ti.add_indicators(df.copy())
        print("add_indicators executed successfully.")
        print("Columns:", df_processed.columns.tolist())

        # Check specific problematic columns
        if "vp_low" in df_processed.columns:
            print("VP columns present.")
        else:
            print("WARNING: VP columns missing or calculation failed silently.")

    except Exception as e:
        print(f"Error in add_indicators: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    test_indicators()
