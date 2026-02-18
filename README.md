# Crypto Trading Bot

An open-source cryptocurrency trading bot framework designed for modularity and extensibility. This project aims to provide a robust pipeline for data acquisition, technical analysis, model training, and live trading execution.

## Installation

To get started, you'll need **Conda** or **Miniconda** installed on your system.

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd <repository_name>
    ```

2.  **Create the Conda environment:**
    This project uses an `environment.yml` file to manage dependencies.
    ```bash
    conda env create -f environment.yml
    ```

3.  **Activate the environment:**
    ```bash
    conda activate crypto_bot
    ```

4.  **Install Jupyter Kernel (Optional):**
    If you plan to use Jupyter Notebooks for experimentation or development, register the kernel so it appears in your notebook interface.
    ```bash
    python -m ipykernel install --user --name=trading-bot --display-name="Trading Bot"
    ```

## Project Status

We are building this bot in stages. Here is the current progress:

-   ✅ **Data Loading**: Used <code>ccxt</code> to download candle data for various symbols over different timeframes.
-   ✅ **Technical Analysis and Normalization**: Added calculation of various technical indicators, and their normalization.
-   ✅ **Trade Execution Logic**: Added the main bot loop and the interface for the bot to interact with the exchange.
-   🚧 **Model Training**: *IN PROGRESS*: We are developing machine learning models for signal generation.
-   🚧 **Model Optimization**: *TODO*: Tune the various hyperparameters to select the best model for bactesting.
-   🚧 **Backtesting**: *TODO*: Backtest the model on past data.
-   ⏳ **Live Trading**: *TODO*: Make the bot easy to access and UI for bot statistics and analysis.

## Usage

The project is controlled via `main.py` using command-line arguments.

### 1. Data Mode (Data Downloading & Analysis)
Use this mode to download historical data, calculate indicators, and generate correlation heatmaps.

**Basic command:**
```bash
python main.py data
```

**Options:**
-   `--symbol`: Override the default trading pair (e.g., `BTC/USDT`, `ETH/USDT`).
-   `--timeframe`: Override the time interval (e.g., `1h`, `15m`, `1d`).
-   `--days`: Specify the number of lookback days for data fetching.

**Example:**
Download 60 days of ETH/USDT data on a 1-hour timeframe:
```bash
python main.py data --symbol ETH/USDT --timeframe 1h --days 60
```

### 2. Trade Mode (Live Trading)
Use this mode to start the live trading loop. *Note: This feature is under active development.*

**Basic command:**
```bash
python main.py trade
```

**Options:**
-   `--symbol`: Override the default symbol to trade.
-   `--timeframe`: Override the default timeframe.

**Example:**
Start the bot for BTC/USDT on a 5-minute timeframe:
```bash
python main.py trade --symbol BTC/USDT --timeframe 5m
```

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.
