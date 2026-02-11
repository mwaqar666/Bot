import ccxt
import pandas as pd
import time
import os
from datetime import datetime, timedelta

import argparse

def fetch_historical_data(symbol="BTC/USDT:USDT", timeframe="15m", days=365):
    """
    Downloads historical OHLCV data from Binance and saves it to a CSV.
    """
    # Initialize Exchange
    exchange = ccxt.binanceusdm({'enableRateLimit': True})
    
    # Time Calc
    end_time = datetime.now()
    start_time = end_time - timedelta(days=days)
    since = int(start_time.timestamp() * 1000)
    
    all_candles = []
    print(f"Fetching {symbol} ({timeframe}) from {start_time.date()}...")
    
    while True:
        try:
            candles = exchange.fetch_ohlcv(symbol, timeframe, since, limit=1000)
            if not candles:
                break
            
            all_candles.extend(candles)
            since = candles[-1][0] + 1
            
            # Progress Date
            dt = datetime.fromtimestamp(candles[-1][0] / 1000)
            print(f"  > Downloaded to: {dt}")
            
            # Check if we reached end time
            if since >= int(end_time.timestamp() * 1000):
                 break
                 
            time.sleep(0.5) 

        except Exception as e:
            print(f"  Error: {e}")
            break
            
    if not all_candles:
        print("  No data found.")
        return None

    # DataFrame
    df = pd.DataFrame(all_candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    
    # Filename
    safe_symbol = symbol.replace('/', '_').replace(':', '_')
    filename = f"ai_bot/data_engine/{safe_symbol}_{timeframe}.csv"
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    df.to_csv(filename)
    
    print(f"  Success! Saved {len(df)} rows to {filename}")
    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--symbol', type=str, default="BTC/USDT:USDT")
    parser.add_argument('--timeframes', nargs='+', default=["15m"])
    parser.add_argument('--days', type=int, default=365)
    
    args = parser.parse_args()
    
    print(f"Starting Batch Download for {args.symbol}...")
    
    for tf in args.timeframes:
        fetch_historical_data(args.symbol, tf, args.days)
        
    print("\n--- All Downloads Complete ---")
