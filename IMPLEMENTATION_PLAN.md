# AI Quantitative Trading System: Implementation & Audit Plan

## 1. Project Overview
This project is a hybrid **AI-Quantitative Trading System** designed for the Binance Futures market (via `ccxt`). It combines traditional algorithmic indicators (EMA, RSI, ADX) with a Reinforcement Learning (PPO) agent to validate trades. The core philosophy is "Algo proposes, AI confirms," providing a dual-layer decision engine to maximize precision and minimize risk.

---

## 2. File-by-File Analysis & Audit

| File | Status | Description | Issues / Bugs | Improvements |
| :--- | :--- | :--- | :--- | :--- |
| **`config.py`** | 🟢 Stable | Stores configuration constants (API keys, pairs, risk params, indicator settings). | Hardcoded `SYMBOL` limiting multi-pair deployment. | Add environment variable overrides for critical params (Symbol, Timeframe) for flexibility. |
| **`execution.py`** | 🟡 Needs Polish | Handles exchange interactions: Data fetching, Orders, Leverage, Position Management. | `cancel_orders` is aggressive (cancels ALL). `get_position` relies on raw `info` (Binance specific). | add `retries` for network calls. Refactor `get_position` to be more CCXT-agnostic if possible. |
| **`indicators.py`** | 🟢 Stable | Calculates Technical Indicators (EMA, RSI, ADX, ATR). | Modifies DataFrame in-place (efficient but careful with references). | Ensure input DF has sufficient history to avoid `NaN`s at the start of the window. |
| **`strategy.py`** | 🟢 Stable | Defines the algorithmic strategy logic (Trend + Momentum + Volatility). | Logic is hardcoded (conditions). | Move strategy logic into a configuration or class to allow hot-swapping strategies. |
| **`main.py`** | 🟡 Critical | The Orchestrator. Fetches data -> Runs Algo -> Runs AI -> Executes Trades. | `_calculate_position_size` uses `free` balance, ignoring open unrealized PnL (Total Equity better). | Switch to `fetch_balance()['total']['USDT']` for risk calculations. Add logging to file (not just print). |
| **`ai_bot/train_agent.py`** | 🟢 Functional | Script to train the PPO model using `stable-baselines3`. | Hardcoded file paths and training steps (`100,000` might be low/high depending on data). | Add command-line arguments for `timesteps`, `data_path`, and `output_dir`. |
| **`ai_bot/inference.py`** | 🟢 Functional | Wraps the trained model for inference in the main loop. | Re-processes entire DataFrame for every inference step (inefficient but safe). | Cache `FeatureEngineer` state or optimize to process only increment if performance becomes an issue (ok for 15m). |
| **`ai_bot/rl_env/trading_env.py`** | 🟡 Logical | The Gym Environment tailored for Crypto Trading (Custom Reward). | Shorting logic in `step` is simplified (1x leverage simulation). Reward function is basic PnL. | Implement leverage simulation in the Env to match Live params. Add "Sharpe Ratio" component to Reward. |
| **`ai_bot/data_engine/feature_engineer.py`** | 🟢 Excellent | Converts raw OHLCV -> Normalized State Vector for AI. | `_add_time_features` assumes DateTimeIndex. | Ensure robust index checks. Add `Vix` or `Funding Rate` as features (advanced). |

---

## 3. "High Precision" Bot Checklist

To achieve a high-precision trading bot (targeting >90% strictly is statistically anomalous in finance, but aiming for >90% *system reliability* and *high risk-adjusted returns* is feasible), the following must be implemented:

### ✅ Functionality Checklist (Foundation)
- [x] **Hybrid Decision Engine**: Algo Filters + AI Confirmation (Confluence).
- [x] **Dynamic Risk Management**: ATR-based Stops and Take Profits.
- [x] **Position Sizing**: Risk-per-trade (1%) logic implemented.
- [x] **Data Normalization**: Inputs scaled for Neural Network (Log returns, normalized RSI).
- [x] **Look-ahead Bias Prevention**: Rigorous `shift(1)` usage in Feature Engineering.
- [x] **Environment Simulation**: Custom Gym Env for training.

### ⚠️ Reliability & Safety Checklist (Critical for Live)
- [ ] **Equity-Based Sizing**: Switch from `free` balance to `total` equity to prevent over-leveraging during drawdowns.
- [ ] **Execution Retry Logic**: If Binance API times out (502/504), the bot must retry instead of crashing or skipping.
- [ ] **Circuit Breakers**: "Emergency Stop" if the bot loses X% of equity in a day to prevent flash crash liquidations.
- [ ] **Cooldown Periods**: Stop trading for X hours after a sequence of losses to prevent "revenge trading" or chopping.
- [ ] **Latency Management**: Measure time from `candle_close` to `order_submit`. Must be < 3 seconds.
- [ ] **Drift Detection**: Monitor if Live Data distribution diverges from Training Data distribution (Concept Drift).

### 🚀 Optimization & Advanced Models (The "90% Accuracy" Path)
- [ ] **Hyperparameter Tuning**: Optimize PPO specs (Learning Rate, Batch Size, Entropy float).
- [ ] **Reward Engineering**: Change reward from simple `PnL` to `Sortino Ratio` (Penalize downside risk more).
- [ ] **Regime Detection**: Implement a supervisor model (e.g., K-Means or HMM) to detect "Bull/Bear/Sideways" regimes and switch strategies/agents accordingly.
- [ ] **Sentiment Analysis**: Integrate FinBERT or similar to veto trades during high-impact news events (FUD/Hacks).
- [ ] **Volatility Prediction**: Use GARCH or LSTM to forecast *future* volatility instead of relying only on past ATR.
- [ ] **Order Book Features**: (Advanced) Ingest Level 2 data (Bid/Ask imbalance) for short-term directional bias.

---

## 4. Current Action Plan
1.  **Refine `main.py`**: Fix the Balance calculation issue and add Circuit Breakers.
2.  **Enhance `execution.py`**: Add Retry Decorators for API calls.
3.  **Train Robust Model**: Run `train_agent.py` on at least 1 year of data.
4.  **Advanced Logic**: Begin researching Regime Detection (K-Means) integration.
5.  **Paper Trade**: Run `main.py` on Testnet for 1 week.
