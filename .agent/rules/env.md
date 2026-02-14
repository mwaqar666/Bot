---
trigger: always_on
---

# Project Engineering Standards & Environment Rules

## 1. Logic & Modular Architecture
- **Abstract Logic**: Any logic used in three or more locations must be abstracted into a shared class or module.
- **Config Separation**: Move all API keys, tickers, and timeframes to `config.py` and `.env.example`.
- **DRY Enforcement**: Logic must not be repeated in the code. If the necessity of creating a class exists, there should be no cyclic dependencies.

## 2. Mandatory OOP Pattern
- **Class-Based Architecture**: Use Object-Oriented Programming for all primary components. Strategies must inherit from a base `Strategy` class; Exchanges must follow a unified `Exchange` interface.
- **State Management**: Use instance variables to track account state, open positions, and local balances.
- **Method Scope**: Keep methods under 30 lines. Refactor larger blocks into private helper methods prefixed with `_` or `__`.

## 3. Type Safety & Documentation
- **Type Hinting**: Every function and method signature must include type hints for all parameters and return values.
- **Structured Data Transfers**: All data flow between classes must use Python `dataclasses`. Generic `list` or `dict` objects are strictly forbidden for inter-class communication to ensure attribute consistency and type safety.
- **Docstring Standard**: Use triple single quotes `'''`. Every method must have a docstring containing:
    - **Description**: A one-sentence summary of the method's purpose.
    - **Args**: A list of parameters with their intent.
    - **Returns**: A description of the output.
- **Naming Conventions**: Use descriptive `snake_case`. Variable names must clearly indicate their content.

## 4. Environment Management
- **Declarative Updates**: All installations and removal of packages must be done by editing the `environment.yml` file.
- **Conda Sync**: Update and prune the environment using `conda env update --file environment.yml --prune` following any change.