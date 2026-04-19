from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from .base import AccountSnapshot, Broker, Fill, Order


@dataclass
class PaperBrokerConfig:
    starting_cash: float = 100000.0
    fee_bps: float = 6.0
    slippage_bps: float = 2.0


class PaperBroker(Broker):
    """
    Paper broker with immediate fills at mark price + slippage and simple fees.
    Positions are tracked in *qty* (shares).
    """

    def __init__(self, config: PaperBrokerConfig):
        self.cfg = config
        self.cash = float(config.starting_cash)
        self.positions: Dict[str, float] = {}

    def get_account(self, mark_prices: Dict[str, float]) -> AccountSnapshot:
        equity = self.cash
        for sym, qty in self.positions.items():
            px = float(mark_prices.get(sym, 0.0))
            equity += qty * px
        return AccountSnapshot(cash=float(self.cash), positions=dict(self.positions), equity=float(equity))

    def place_order(self, order: Order, mark_price: float, timestamp_ms: int) -> Fill:
        px = float(mark_price)
        qty = float(order.qty)
        if qty <= 0:
            raise ValueError("qty must be > 0")

        slip = self.cfg.slippage_bps / 10000.0
        fee_rate = self.cfg.fee_bps / 10000.0

        # Apply slippage against the trader
        fill_price = px * (1.0 + slip) if order.side == "buy" else px * (1.0 - slip)
        notional = qty * fill_price
        fee = notional * fee_rate

        if order.side == "buy":
            total_cost = notional + fee
            if total_cost > self.cash:
                # Partial fill to available cash
                qty = max(0.0, (self.cash / (fill_price * (1.0 + fee_rate))) if fill_price > 0 else 0.0)
                notional = qty * fill_price
                fee = notional * fee_rate
                total_cost = notional + fee
            self.cash -= total_cost
            self.positions[order.symbol] = self.positions.get(order.symbol, 0.0) + qty
        else:
            held = self.positions.get(order.symbol, 0.0)
            sell_qty = min(held, qty)
            proceeds = sell_qty * fill_price - fee
            self.cash += proceeds
            new_qty = held - sell_qty
            if new_qty <= 0:
                self.positions.pop(order.symbol, None)
            else:
                self.positions[order.symbol] = new_qty
            qty = sell_qty

        return Fill(
            symbol=order.symbol,
            side=order.side,
            qty=float(qty),
            price=float(fill_price),
            fee=float(fee),
            timestamp_ms=int(timestamp_ms),
        )

