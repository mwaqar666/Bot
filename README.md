# Crypto Futures Trading Bot

This bot trades **BTC/USDT** on Binance Futures using a **Trend Following + Momentum** strategy.

## Strategy Summary
*   **Timeframe**: 15 Minutes
*   **Indicators**:
    *   **EMA 9 & 21**: Determine Trend Direction (Bullish/Bearish).
    *   **RSI 14**: Momentum Filter (Avoid buying overbought > 70).
    *   **ADX 14**: Volatility Filter (Only trade if Trend Strength > 25).
    *   **ATR 14**: Dynamic Stop Loss calculation.

## Configuration
All settings are in `config.py`.
*   **Symbol**: BTC/USDT
*   **Leverage**: 10x
*   **Risk Per Trade**: 1% of account balance.

## Installation
1.  **Install Conda** (if not already installed).
2.  **Create Environment**:
    ```bash
    conda create -n crypto_bot python=3.10 -y
    conda activate crypto_bot
    conda install -c conda-forge ccxt pandas pandas-ta python-dotenv schedule
    ```
3.  **Setup Keys**:
    *   Rename `.env.example` to `.env`.
    *   Add your Binance API Key & Secret (Use Testnet keys for safety!).

## Usage
Run the bot:
```bash
python main.py
```

## Disclaimer
Trading futures involves significant risk. This bot is for educational purposes only. Use at your own risk.
