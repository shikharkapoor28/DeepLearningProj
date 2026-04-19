from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Literal, Optional


Side = Literal["buy", "sell"]


@dataclass(frozen=True)
class Order:
    symbol: str
    side: Side
    qty: float
    limit_price: Optional[float] = None


@dataclass(frozen=True)
class Fill:
    symbol: str
    side: Side
    qty: float
    price: float
    fee: float
    timestamp_ms: int


@dataclass(frozen=True)
class AccountSnapshot:
    cash: float
    positions: Dict[str, float]  # symbol -> qty
    equity: float


class Broker:
    """
    Minimal broker interface (paper or live).
    """

    def get_account(self, mark_prices: Dict[str, float]) -> AccountSnapshot:
        raise NotImplementedError

    def place_order(self, order: Order, mark_price: float, timestamp_ms: int) -> Fill:
        raise NotImplementedError

