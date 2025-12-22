"""Rule class for the GA trading system."""

import random
from dataclasses import dataclass
from typing import Any

import pandas as pd

from jarvis.ga.indicators import Indicator, indicator_from_dict, random_indicator


@dataclass
class Rule:
    """A trading rule that combines an indicator with a target and weight.

    The contribution is calculated as: (value - target) * weight
    Where value is the indicator's calculated value.

    Positive weight + value > target = positive contribution (buy signal)
    Negative weight + value > target = negative contribution (sell signal)
    """

    indicator: Indicator
    target: float
    weight: float

    def calculate_contribution(self, df: pd.DataFrame) -> float:
        """Calculate this rule's contribution to the signal.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            (indicator_value - target) * weight, or 0.0 if NaN
        """
        import math

        value = self.indicator.calculate(df)
        if math.isnan(value):
            return 0.0
        return (value - self.target) * self.weight

    def mutate(self) -> "Rule":
        """Return a mutated copy of this rule.

        Randomly mutates one of: indicator, target, or weight.
        """
        mutation_type = random.choice(["indicator", "target", "weight"])

        if mutation_type == "indicator":
            # Mutate indicator parameters or replace with new type
            if random.random() < 0.3:
                # 30% chance to replace indicator type
                new_indicator = random_indicator()
            else:
                # 70% chance to mutate parameters
                new_indicator = self.indicator.mutate()
            return Rule(indicator=new_indicator, target=self.target, weight=self.weight)

        elif mutation_type == "target":
            # Mutate target by a small factor
            factor = random.uniform(0.9, 1.1)
            new_target = self.target * factor
            return Rule(indicator=self.indicator, target=new_target, weight=self.weight)

        else:  # weight
            # Mutate weight by a small amount
            delta = random.uniform(-0.1, 0.1)
            new_weight = self.weight + delta
            return Rule(indicator=self.indicator, target=self.target, weight=new_weight)

    @classmethod
    def random(cls, price_hint: float | None = None) -> "Rule":
        """Create a random rule.

        Args:
            price_hint: Approximate current price of the asset.
                        Used to set reasonable target ranges for price-based indicators.
                        If None, uses a default range suitable for BTC.
        """
        indicator = random_indicator()

        # Default price hint for backwards compatibility
        if price_hint is None:
            price_hint = 50000.0  # BTC-ish default

        # Set initial target based on indicator type
        indicator_type = indicator.to_dict()["type"]
        if indicator_type == "RSI":
            target = random.uniform(20, 80)
        elif indicator_type in ("MACD", "MACD_HIST"):
            # MACD values scale with price, use percentage of price
            target = random.uniform(-0.02, 0.02) * price_hint
        elif indicator_type == "VOLUME":
            target = random.uniform(100000, 10000000)
        elif indicator_type == "PRICE":
            # Target around current price (+/- 20%)
            target = random.uniform(0.8, 1.2) * price_hint
        else:  # SMA, EMA
            # Moving averages around current price (+/- 20%)
            target = random.uniform(0.8, 1.2) * price_hint

        weight = random.uniform(-1, 1)

        return cls(indicator=indicator, target=target, weight=weight)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "indicator": self.indicator.to_dict(),
            "target": self.target,
            "weight": self.weight,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Rule":
        """Deserialize from dictionary."""
        indicator = indicator_from_dict(data["indicator"])
        return cls(
            indicator=indicator,
            target=data["target"],
            weight=data["weight"],
        )

    def __str__(self) -> str:
        """Human-readable rule description."""
        ind_dict = self.indicator.to_dict()
        ind_type = ind_dict["type"]
        params = {k: v for k, v in ind_dict.items() if k != "type"}
        params_str = ",".join(f"{v}" for v in params.values()) if params else ""
        sign = "+" if self.weight > 0 else ""
        return f"{ind_type}({params_str}) > {self.target:.2f} * {sign}{self.weight:.3f}"
