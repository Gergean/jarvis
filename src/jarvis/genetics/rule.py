"""Rule class for the GA trading system."""

import random
from dataclasses import dataclass
from typing import Any

from jarvis.genetics.indicators import OHLCV, Indicator, indicator_from_dict, random_indicator


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

    # Normalization constant for contribution calculation
    WEIGHT_SCALE = 100_000

    def calculate_contribution(self, ohlcv: OHLCV) -> float:
        """Calculate this rule's contribution to the signal.

        Args:
            ohlcv: OHLCV named tuple with numpy arrays

        Returns:
            (indicator_value - target) * weight / WEIGHT_SCALE, or 0.0 if NaN
        """
        import math

        value = self.indicator.calculate(ohlcv)
        if math.isnan(value):
            return 0.0
        return (value - self.target) * self.weight / self.WEIGHT_SCALE

    def calculate_contribution_detailed(self, ohlcv: OHLCV) -> tuple[float, float, float]:
        """Calculate contribution with detailed breakdown for debugging.

        Args:
            ohlcv: OHLCV named tuple with numpy arrays

        Returns:
            Tuple of (indicator_value, target, contribution)
            If value is NaN, returns (NaN, target, 0.0)
        """
        import math

        value = self.indicator.calculate(ohlcv)
        if math.isnan(value):
            return (float("nan"), self.target, 0.0)
        contribution = (value - self.target) * self.weight / self.WEIGHT_SCALE
        return (value, self.target, contribution)

    def mutate(self, interval: str = "1h") -> "Rule":
        """Return a mutated copy of this rule.

        Randomly mutates one of: indicator, target, or weight.
        """
        mutation_type = random.choice(["indicator", "target", "weight"])

        if mutation_type == "indicator":
            # Mutate indicator parameters or replace with new type
            if random.random() < 0.3:
                # 30% chance to replace indicator type
                new_indicator = random_indicator(interval)
            else:
                # 70% chance to mutate parameters
                new_indicator = self.indicator.mutate(interval)
            return Rule(indicator=new_indicator, target=self.target, weight=self.weight)

        elif mutation_type == "target":
            # Mutate target within indicator's valid range
            min_val, max_val = self.indicator.get_range()
            # Small random step within range
            range_size = max_val - min_val
            step = random.uniform(-0.1, 0.1) * range_size
            new_target = max(min_val, min(max_val, self.target + step))
            return Rule(indicator=self.indicator, target=new_target, weight=self.weight)

        else:  # weight
            # Mutate weight by a percentage (10-20% change)
            factor = random.uniform(0.8, 1.2)
            new_weight = self.weight * factor
            # Occasionally flip sign
            if random.random() < 0.1:
                new_weight = -new_weight
            return Rule(indicator=self.indicator, target=self.target, weight=new_weight)

    @classmethod
    def random(cls, price_hint: float | None = None, interval: str = "1h") -> "Rule":
        """Create a random rule.

        Args:
            price_hint: Not used anymore (kept for backwards compatibility).
            interval: Trading interval for period calculation (e.g., "1h", "4h", "1d").
        """
        indicator = random_indicator(interval)

        # Get target from indicator's valid range
        min_val, max_val = indicator.get_range()
        target = random.uniform(min_val, max_val)

        weight = random.uniform(-1_000_000, 1_000_000)

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
