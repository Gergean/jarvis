"""Individual class for the GA trading system."""

import copy
import random
from dataclasses import dataclass, field
from typing import Any

from jarvis.genetics.indicators import OHLCV
from jarvis.genetics.rule import Rule
from jarvis.models import ActionType, PositionSide


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

        total = sum(rule.calculate_contribution(ohlcv) for rule in self.rules)

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
