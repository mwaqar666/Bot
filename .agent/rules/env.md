---
trigger: always_on
---

# Python & Conda Execution Rules

## 1. Path & Environment Management
- **Shell**: ALWAYS use PowerShell 7 (`pwsh.exe`).
- **Conda Path**: Use full path for environment management: `C:\Users\mwaqa\anaconda3\Scripts\conda.exe`.
- **Python Path**: Use the specific project interpreter for ALL runs/installs: `C:\Users\mwaqa\anaconda3\envs\crypto_bot\python.exe`.
- **Execution Policy**: 
    1. NEVER use `python` or `conda` as bare commands.
    2. ALWAYS use the full paths above with the call operator `&` for execution (e.g., `& "C:\...\python.exe" main.py`).

## 2. Code Structure & DRY (Don't Repeat Yourself)
- **Rule of Three**: If the same logic (e.g., fetching a candle or calculating RSI) appears three times, it MUST be abstracted into a shared function or class.
- **Centralized Config**: Hardcoded values (API keys, Symbols, Timeframes) are strictly forbidden in logic files. They must live in `config.py` or a `.env` file.
- **Single Representation**: Every piece of knowledge (like reward function logic) must exist in exactly one place to prevent "logic drift" between training and live trading.

## 3. Anti-God Function Policy
- **One Task, One Function**: Each function must do one thing and one thing well (e.g., `fetch_ohlcv` only fetches data; it does not also calculate indicators).
- **Size Limits**: Keep methods under 50 lines. If a function is longer than the screen height, refactor it into smaller sub-functions.
- **Separation of Concerns**: Keep **Execution** (API), **Strategy** (logic), and **AI** (inference) in separate modules. `main.py` should only act as a high-level orchestrator.

## 4. Documentation & Type Safety
- **Mandatory Type Hinting**: All function signatures must include type hints for parameters and return values.
    - *Example*: `def get_balance(symbol: str) -> float:`
- **Action-Oriented Docstrings**: Use triple single quotes `'''` for docstrings. Describe the **why** and the **returns**, not just a play-by-play of the code.
- **Self-Documenting Names**: Use descriptive `lower_case_with_underscores`. A function named `check_rsi_overbought()` is required over generic names like `check_val()`.