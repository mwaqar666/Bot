from typing import Optional

from framework.data.data_types import Position, Order

import ccxt
import pandas as pd


class TradeExecutor:
    def __init__(self, api_key: str, api_secret: str, testnet: bool = True) -> None:
        """
        Initializes the connection to the Binance Futures exchange.
        """
        self.exchange = ccxt.binanceusdm(
            {
                "apiKey": api_key,
                "secret": api_secret,
                "enableRateLimit": True,
                "options": {
                    "defaultType": "future",  # Must be 'future' to trade Futures
                    "adjustForTimeDifference": True,
                },
            }
        )
        self.exchange.enable_demo_trading(testnet)  # Use Testnet for paper trading
        print(f"Connected to Binance Futures ({'TESTNET' if testnet else 'LIVE'})")

    def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int = 100) -> pd.DataFrame:
        """
        Fetches historical candlestick data (Open, High, Low, Close, Volume).
        """
        try:
            # Fetch bars
            bars = self.exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)

            # Convert to DataFrame
            df = pd.DataFrame(bars, columns=["timestamp", "open", "high", "low", "close", "volume"])
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")

            return df
        except Exception as e:
            print(f"Error fetching data for {symbol}: {e}")
            return pd.DataFrame()

    def get_balance(self, asset: str = "USDT") -> float:
        """
        Returns the free balance of the specified asset (usdt usually).
        """
        try:
            balance = self.exchange.fetch_balance()
            return float(balance[asset]["free"])
        except Exception as e:
            print(f"Error fetching balance: {e}")
            return 0.0

    def set_leverage(self, symbol: str, leverage: int) -> None:
        """
        Sets the leverage for the given symbol (e.g., BTC/USDT).
        """
        try:
            # Different exchanges have slightly different API calls for leverage
            # CCXT unifies this mostly but verify for Binance
            self.exchange.set_leverage(leverage, symbol)
            print(f"Leverage set to {leverage}x for {symbol}")
        except Exception as e:
            print(f"Error setting leverage: {e}")

    def place_order(self, symbol: str, side: str, amount: float, price: Optional[float] = None, stop_loss: Optional[float] = None, take_profit: Optional[float] = None) -> Optional[Order]:
        """
        Places an order with optional Stop Loss and Take Profit.
        Orchestrates 3 separate orders for Binance Futures.
        """
        try:
            if price:
                order_type = "limit"
            else:
                order_type = "market"

            print(f"Placing ENTRY {side.upper()} {amount} {symbol}...")
            entry_order = self.exchange.create_order(symbol, order_type, side, amount, price)
            print(f"Entry Order Placed: {entry_order['id']}")

            # 2. Place Stop Loss (if provided)
            if stop_loss:
                sl_side = "sell" if side == "buy" else "buy"
                sl_params = {"stopPrice": stop_loss, "reduceOnly": True}

                print(f"Placing STOP LOSS at {stop_loss}...")
                self.exchange.create_order(symbol, "STOP_MARKET", sl_side, amount, None, sl_params)

            # 3. Place Take Profit (if provided)
            if take_profit:
                tp_side = "sell" if side == "buy" else "buy"
                tp_params = {
                    "stopPrice": take_profit,
                    "reduceOnly": True,
                }  # Binance uses stopPrice for TP too often

                print(f"Placing TAKE PROFIT at {take_profit}...")
                self.exchange.create_order(symbol, "TAKE_PROFIT_MARKET", tp_side, amount, None, tp_params)

            # Return standard Order object
            return Order(
                id=str(entry_order["id"]),
                symbol=symbol,
                side=side,  # type: ignore
                amount=float(entry_order.get("amount", amount)),
                price=(float(entry_order.get("price", price)) if entry_order.get("price") else price),
                status=entry_order.get("status", "unknown"),
                type=order_type,
            )

        except Exception as e:
            print(f"Error placing order chain: {e}")
            return None

    def cancel_orders(self, symbol: str) -> None:
        """Cancels all open orders for a symbol (SL/TP)"""
        try:
            self.exchange.cancel_all_orders(symbol)
            print(f"Cancelled all open orders for {symbol}")
        except Exception as e:
            print(f"Error cancelling orders: {e}")

    def get_position(self, symbol: str) -> Optional[Position]:
        """
        Check if we have an open position for the symbol.
        Returns the Position object or None.
        """
        try:
            positions = self.exchange.fetch_positions(symbols=[symbol])

            for position in positions:
                # CCXT usually returns a list. We filter for the specific symbol.
                # Check if size is non-zero (meaning open)
                # Binance Futures specific: contracts is absolute size
                size = float(position.get("contracts", 0))

                if position["symbol"] == symbol and size > 0:
                    side = position["side"]

                    # Double check with positionAmt if available (Binance specific)
                    # If positionAmt is negative, it is a SHORT
                    if "info" in position and "positionAmt" in position["info"]:
                        amt = float(position["info"]["positionAmt"])
                        if amt < 0:
                            side = "short"
                        elif amt > 0:
                            side = "long"

                    return Position(
                        symbol=symbol,
                        side=side,  # type: ignore
                        amount=size,
                        entry_price=float(position["entryPrice"]),
                        unrealized_pnl=float(position["unrealizedPnl"]),
                    )
            return None
        except Exception as e:
            # Often on testnet or empty account, fetching positions might fail or return empty
            print(f"Debug: No open position found or error: {e}")
            return None
