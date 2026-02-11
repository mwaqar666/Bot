# CryptoTradingEnv: Documentation & Analysis

## Overview
`CryptoTradingEnv` is a custom Reinforcement Learning environment compatible with OpenAI Gym / Gymnasium. It simulates a cryptocurrency exchange, allowing an agent to learn trading strategies using historical OHLCV data.

### Key Features
1.  **Action Space**: Discrete(3) -> `0: Hold`, `1: Buy (Long)`, `2: Sell (Short)`.
2.  **Observation Space**: A sliding window of historical data `(Window_Size, Num_Features)`. This is designed for sequence models like Transformers.
3.  **Intra-Candle Simulation**: The environment checks the `High` and `Low` of the *current* candle to determine if a Stop Loss (SL) or Take Profit (TP) was hit *before* processing the agent's action for that step.
4.  **Fee Simulation**: A percentage fee (default 0.1%) is deducted from the Net Worth upon every trade entry and exit.
5.  **Dynamic Stops**: SL and TP levels are calculated dynamically based on the ATR (Average True Range) volatility at the time of entry.
6.  **Performance Tracking**: Maintains a `trade_history` list containing entry/exit prices, PnL, and reasons for exit.

---

## Logic Flow (Step Function)
1.  **Advance Time**: `self.current_step += 1`.
2.  **Fetch Data**: Gets the OHLCV + ATR for the *new* step.
3.  **Intra-Candle Check**:
    *   If `Low <= SL` -> Close Long (SL).
    *   If `High >= TP` -> Close Long (TP).
    *   (Inverse for Short).
    *   If a stop is triggered, the position is closed *before* the agent can act.
4.  **Agent Action**:
    *   If *not* stopped out in this step, execute Agent's command (Buy/Sell/Hold).
    *   If Agent reverses position (e.g., Long -> Short), it counts as 2 executions (Close + Open).
5.  **Update Valuation**: Net Worth is recalculated based on the new `Close` price.
6.  **Reward Calculation**:
    *   `Reward = % Change in Net Worth`.
    *   `Penalty = -0.05 * Number of Executions`.
7.  **Termination Check**:
    *   `Terminated` if Net Worth < 50% of Initial Balance (Bankruptcy).
    *   `Truncated` if `current_step >= max_steps` (End of Data).

---

## Potential Issues & Improvements

### 1. The "Blind Step" Risk
*   **Issue**: The `step()` method increments `self.current_step += 1` at the very beginning without checking if it's already at `max_steps`.
*   **Risk**: If the training loop (e.g., SB3) accidentally calls `step()` one more time after `truncated=True`, the environment will crash with `IndexError`.
*   **Mitigation**: Rely on the training framework to respect the `truncated` flag. A safety guard was discussed but removed to keep logic simple.

### 2. Execution Price Assumption
*   **Issue**: When the Agent decides to Buy/Sell, the trade is executed at the **Close** price of the current candle (implied by `current_price = data["close"]`).
*   **Reality**: In live trading, a decision made *after* observing a candle is usually executed at the **Open** of the *next* candle.
*   **Impact**: This introduces a slight "Lookahead Bias" regarding the specific execution price (getting the Close price knowing the Close price).
*   **Improvement**: Execute trades at `data["open"]` of `step + 1` (requires logic shift) OR accept that `Close` is a proxy for "Instant Execution" in a simulation.

### 3. Intra-Candle Logic Order
*   **Issue**: We check Stops (SL/TP) *before* the Agent acts.
*   **Implication**: If a stop is hit, the Agent's action for that step is ignored (`if not stop_triggered: execute_action`). This assumes the stop happened "during" the candle and the agent realizes it "after" the candle. This is generally a fair assumption for 15m/1h candles.

---

## Checklist: Before Live Deployment

- [ ] **Data Validation**: Ensure the input DataFrame `df` is sorted by timestamp ascending.
- [ ] **Feature Consistency**: Verify that `features` list matches exactly what the `FeatureEngineer` produces (columns exist).
- [ ] **Fee Alignment**: Double-check `fee_percent`. Binance standard is 0.1% (`0.001`), but BNB holders get 0.075%.
- [ ] **Reward Tuning**: Monitor if the `-0.05` penalty is too harsh or too lenient. Does the bot just "Hold" forever to avoid fees?
- [ ] **Metric Validation**: Compare the `render()` output Profit vs. a manual calculation of the `trade_history` PnL to ensure accuracy.
