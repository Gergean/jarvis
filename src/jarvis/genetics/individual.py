"""Individual class for the GA trading system."""

import copy
import logging
import random
from dataclasses import dataclass, field
from typing import Any

from jarvis.genetics.indicators import OHLCV
from jarvis.genetics.rule import Rule
from jarvis.models import ActionType, PositionSide

logger = logging.getLogger("jarvis")


@dataclass
class Individual:
    """An individual trading strategy composed of multiple rules.

    Signal calculation:
    1. Calculate contribution from each rule: (value - target) * weight
    2. Sum all contributions
    3. Position-aware thresholds:
       - total > 1 and no position -> LONG
       - total < -1 and no position -> SHORT
       - total < -0.5 and in LONG -> CLOSE
       - total > 0.5 and in SHORT -> CLOSE
       - Otherwise -> STAY
    """

    rules: list[Rule] = field(default_factory=list)
    fitness: float = 0.0

    # Thresholds for signal generation
    LONG_THRESHOLD = 1.0
    SHORT_THRESHOLD = -1.0
    CLOSE_THRESHOLD = 0.5

    def get_total_score(self, ohlcv: OHLCV) -> float:
        """Calculate total score from all rules.

        Args:
            ohlcv: OHLCV named tuple with numpy arrays

        Returns:
            Sum of all rule contributions (positive = bullish, negative = bearish)
        """
        if not self.rules:
            return 0.0
        return sum(rule.calculate_contribution(ohlcv) for rule in self.rules)

    def get_signal(self, ohlcv: OHLCV, current_side: PositionSide = PositionSide.NONE) -> ActionType:
        """Calculate trading signal from rules with position awareness.

        Args:
            ohlcv: OHLCV named tuple with numpy arrays
            current_side: Current position direction (LONG, SHORT, or NONE)

        Returns:
            ActionType.LONG, ActionType.SHORT, ActionType.CLOSE, or ActionType.STAY
        """
        if not self.rules:
            return ActionType.STAY

        total = self.get_total_score(ohlcv)


        # No position - can open new
        if current_side == PositionSide.NONE:
            if total > self.LONG_THRESHOLD:
                return ActionType.LONG
            elif total < self.SHORT_THRESHOLD:
                return ActionType.SHORT
            else:
                return ActionType.STAY

        # In long position
        elif current_side == PositionSide.LONG:
            if total < -self.CLOSE_THRESHOLD:
                return ActionType.CLOSE
            else:
                return ActionType.STAY

        # In short position
        elif current_side == PositionSide.SHORT:
            if total > self.CLOSE_THRESHOLD:
                return ActionType.CLOSE
            else:
                return ActionType.STAY

        return ActionType.STAY

    def mutate(self, mutation_rate: float = 0.1) -> "Individual":
        """Return a mutated copy of this individual.

        Args:
            mutation_rate: Probability of mutating each rule

        Returns:
            A new Individual with mutated rules
        """
        new_rules = []
        for rule in self.rules:
            if random.random() < mutation_rate:
                new_rules.append(rule.mutate())
            else:
                new_rules.append(copy.deepcopy(rule))

        # Small chance to add or remove a rule
        if random.random() < 0.05 and len(new_rules) > 1:
            # Remove a random rule
            new_rules.pop(random.randint(0, len(new_rules) - 1))
        elif random.random() < 0.05:
            # Add a new random rule
            new_rules.append(Rule.random())

        return Individual(rules=new_rules, fitness=0.0)

    @classmethod
    def crossover(cls, parent1: "Individual", parent2: "Individual") -> "Individual":
        """Create a child by combining rules from two parents.

        Uses uniform crossover: each rule is randomly taken from either parent.
        """
        child_rules = []

        # Take rules from both parents
        max_rules = max(len(parent1.rules), len(parent2.rules))
        for i in range(max_rules):
            if i < len(parent1.rules) and i < len(parent2.rules):
                # Both parents have this position, randomly choose
                if random.random() < 0.5:
                    child_rules.append(copy.deepcopy(parent1.rules[i]))
                else:
                    child_rules.append(copy.deepcopy(parent2.rules[i]))
            elif i < len(parent1.rules):
                # Only parent1 has this rule
                if random.random() < 0.5:
                    child_rules.append(copy.deepcopy(parent1.rules[i]))
            else:
                # Only parent2 has this rule
                if random.random() < 0.5:
                    child_rules.append(copy.deepcopy(parent2.rules[i]))

        # Ensure at least one rule
        if not child_rules:
            child_rules.append(Rule.random())

        return cls(rules=child_rules, fitness=0.0)

    @classmethod
    def random(cls, num_rules: int = 5, price_hint: float | None = None) -> "Individual":
        """Create a random individual with the specified number of rules.

        Args:
            num_rules: Number of rules to generate
            price_hint: Approximate price of the asset for setting target ranges
        """
        rules = [Rule.random(price_hint=price_hint) for _ in range(num_rules)]
        return cls(rules=rules, fitness=0.0)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary for JSON export."""
        return {
            "rules": [rule.to_dict() for rule in self.rules],
            "fitness": self.fitness,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Individual":
        """Deserialize from dictionary."""
        rules = [Rule.from_dict(r) for r in data["rules"]]
        return cls(rules=rules, fitness=data.get("fitness", 0.0))

    def __repr__(self) -> str:
        return f"Individual(rules={len(self.rules)}, fitness={self.fitness:.2f})"

    def to_pine_script(self, strategy_name: str = "GA Strategy") -> str:
        """Generate Pine Script code for TradingView.

        Args:
            strategy_name: Name for the strategy

        Returns:
            Pine Script v5 code as string
        """
        lines = [
            "//@version=5",
            f'strategy("{strategy_name}", overlay=true, initial_capital=100, default_qty_type=strategy.percent_of_equity, default_qty_value=20)',
            "",
            "// === INDICATORS ===",
        ]

        # Track unique indicators to avoid duplicates
        indicator_vars = {}  # (type, params_tuple) -> var_name
        var_counter = {}
        indicator_var_names = []  # Track all indicator var names for na check

        for rule in self.rules:
            ind_dict = rule.indicator.to_dict()
            ind_type = ind_dict["type"]
            params = ind_dict.get("params", {})

            # Create unique key for this indicator
            params_tuple = tuple(sorted(params.items()))
            key = (ind_type, params_tuple)

            if key not in indicator_vars:
                var_counter[ind_type] = var_counter.get(ind_type, 0) + 1
                var_name = f"{ind_type.lower()}_{var_counter[ind_type]}"

                if ind_type == "SMA":
                    period = params.get("period", 20)
                    lines.append(f"{var_name} = ta.sma(close, {period})")
                    indicator_var_names.append(var_name)
                elif ind_type == "EMA":
                    period = params.get("period", 20)
                    lines.append(f"{var_name} = ta.ema(close, {period})")
                    indicator_var_names.append(var_name)
                elif ind_type == "RSI":
                    period = params.get("period", 14)
                    lines.append(f"{var_name} = ta.rsi(close, {period})")
                    indicator_var_names.append(var_name)
                elif ind_type == "MACD":
                    fast = params.get("fast", 12)
                    slow = params.get("slow", 26)
                    signal = params.get("signal", 9)
                    lines.append(f"[{var_name}, {var_name}_sig, {var_name}_hist] = ta.macd(close, {fast}, {slow}, {signal})")
                    indicator_var_names.append(var_name)
                elif ind_type == "MACD_HIST":
                    fast = params.get("fast", 12)
                    slow = params.get("slow", 26)
                    signal = params.get("signal", 9)
                    lines.append(f"[{var_name}_m, {var_name}_s, {var_name}] = ta.macd(close, {fast}, {slow}, {signal})")
                    indicator_var_names.append(var_name)
                elif ind_type == "PRICE":
                    var_name = "close"
                elif ind_type == "VOLUME":
                    var_name = "volume"

                indicator_vars[key] = var_name

        # Replace NaN with default values (matching Python behavior)
        lines.append("")
        lines.append("// === NaN HANDLING (match Python fallback values) ===")
        for var_name in indicator_var_names:
            # Python uses 0.0 for SMA/EMA/MACD when nan, 50.0 for RSI
            if var_name.startswith("rsi"):
                lines.append(f"{var_name} := na({var_name}) ? 50.0 : {var_name}")
            else:
                lines.append(f"{var_name} := na({var_name}) ? 0.0 : {var_name}")

        lines.append("")
        lines.append("// === SIGNAL CALCULATION ===")
        lines.append("// Formula: score = sum((indicator_value - target) * weight / 100000)")
        lines.append("score = 0.0")

        for rule in self.rules:
            ind_dict = rule.indicator.to_dict()
            ind_type = ind_dict["type"]
            params = ind_dict.get("params", {})
            params_tuple = tuple(sorted(params.items()))
            key = (ind_type, params_tuple)

            var_name = indicator_vars[key]
            target = rule.target
            weight = rule.weight

            # Normalize by WEIGHT_SCALE (same as Python)
            lines.append(f"score := score + ({var_name} - {target:.6f}) * {weight:.2f} / 100000.0")

        # Find max lookback period needed
        max_period = 200  # Default safety margin
        for rule in self.rules:
            ind_dict = rule.indicator.to_dict()
            params = ind_dict.get("params", {})
            period = params.get("period", 0)
            slow = params.get("slow", 0)  # For MACD
            max_period = max(max_period, period, slow)

        lines.extend([
            "",
            "// === STRATEGY LOGIC ===",
            f"longThreshold = {self.LONG_THRESHOLD}",
            f"shortThreshold = {self.SHORT_THRESHOLD}",
            f"closeThreshold = {self.CLOSE_THRESHOLD}",
            f"minBars = {max_period}  // Warmup period for indicators",
            "",
            "// Track position state",
            "var int positionState = 0  // 0=none, 1=long, -1=short",
            "",
            "// Skip first N bars (indicator warmup period)",
            "if bar_index >= minBars",
            "    // Open Long (only if no position)",
            "    if score > longThreshold and positionState == 0",
            '        strategy.entry("Long", strategy.long)',
            "        positionState := 1",
            "",
            "    // Open Short (only if no position)",
            "    if score < shortThreshold and positionState == 0",
            '        strategy.entry("Short", strategy.short)',
            "        positionState := -1",
            "",
            "    // Close Long (if in long and score drops)",
            "    if score < -closeThreshold and positionState == 1",
            '        strategy.close("Long")',
            "        positionState := 0",
            "",
            "    // Close Short (if in short and score rises)",
            "    if score > closeThreshold and positionState == -1",
            '        strategy.close("Short")',
            "        positionState := 0",
            "",
            "// === PLOTS ===",
            'plot(score, color=score > 0 ? color.green : color.red, title="Score")',
            'hline(longThreshold, "Long", color=color.green, linestyle=hline.style_dotted)',
            'hline(shortThreshold, "Short", color=color.red, linestyle=hline.style_dotted)',
            'bgcolor(bar_index < minBars ? color.new(color.gray, 90) : na, title="Warmup")',
        ])

        return "\n".join(lines)
