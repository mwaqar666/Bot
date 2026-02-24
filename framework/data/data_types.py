from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime
from enum import Enum


class PositionSide(int, Enum):
    LONG = 0
    SHORT = 1


class OrderSide(int, Enum):
    BUY = 0
    SELL = 1


class OrderType(str, Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP_MARKET = "STOP_MARKET"
    TAKE_PROFIT_MARKET = "TAKE_PROFIT_MARKET"


class Action(int, Enum):
    BUY = 2
    SELL = 0
    HOLD = 1


class SignalDirection(int, Enum):
    BUY = 0
    SELL = 1
    HOLD = 2


@dataclass
class Position:
    symbol: str
    side: PositionSide
    amount: float
    entry_price: float
    stop_loss: float
    take_profit: float


@dataclass
class Order:
    id: str
    symbol: str
    side: OrderSide
    amount: float
    price: Optional[float]
    status: str
    type: OrderType = OrderType.MARKET
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class Trade:
    symbol: str
    entry_price: float
    exit_price: float
    amount: float
    side: PositionSide
    pnl: float
    entry_time: datetime
    exit_time: datetime
    reason: str = "Signal"


@dataclass
class TradeSignal:
    direction: SignalDirection
    price: float
    stop_loss: float
    take_profit: float
    reason: str
    confidence: float = 1.0
