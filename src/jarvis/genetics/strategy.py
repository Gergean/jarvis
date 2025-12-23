"""Strategy and TestResult models for GA trading system."""

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from jarvis.genetics.individual import Individual


@dataclass
class TrainingConfig:
    """Configuration used during training."""

    interval: str
    start_date: str
    end_date: str
    generations: int
    population_size: int
    rules_per_individual: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "interval": self.interval,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "generations": self.generations,
            "population_size": self.population_size,
            "rules_per_individual": self.rules_per_individual,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TrainingConfig":
        return cls(
            interval=data["interval"],
            start_date=data["start_date"],
            end_date=data["end_date"],
            generations=data["generations"],
            population_size=data["population_size"],
            rules_per_individual=data.get("rules_per_individual", 5),
        )


@dataclass
class Strategy:
    """A trained trading strategy for a specific symbol.

    Strategy contains only the rules and training metadata.
    Test results are stored separately in TestResult.
    """

    id: str
    symbol: str
    individual: Individual
    training: TrainingConfig
    created_at: datetime = field(default_factory=datetime.utcnow)

    @classmethod
    def create(
        cls,
        symbol: str,
        individual: Individual,
        training: TrainingConfig,
    ) -> "Strategy":
        """Create a new strategy with auto-generated ID."""
        # Generate short unique ID
        short_id = uuid.uuid4().hex[:8]
        strategy_id = f"{symbol}_{short_id}"
        return cls(
            id=strategy_id,
            symbol=symbol,
            individual=individual,
            training=training,
            created_at=datetime.utcnow(),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "symbol": self.symbol,
            "created_at": self.created_at.isoformat(),
            "training": self.training.to_dict(),
            "individual": self.individual.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Strategy":
        return cls(
            id=data["id"],
            symbol=data["symbol"],
            created_at=datetime.fromisoformat(data["created_at"]),
            training=TrainingConfig.from_dict(data["training"]),
            individual=Individual.from_dict(data["individual"]),
        )

    def save(self, directory: str | Path = "strategies") -> Path:
        """Save strategy to JSON file."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        filepath = directory / f"{self.id}.json"

        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

        return filepath

    @classmethod
    def load(cls, filepath: str | Path) -> "Strategy":
        """Load strategy from JSON file."""
        with open(filepath) as f:
            data = json.load(f)

        return cls.from_dict(data)

    @classmethod
    def load_by_id(cls, strategy_id: str, directory: str | Path = "strategies") -> "Strategy":
        """Load strategy by ID."""
        filepath = Path(directory) / f"{strategy_id}.json"
        return cls.load(filepath)

    def __repr__(self) -> str:
        return f"Strategy({self.id}, rules={len(self.individual.rules)})"


@dataclass
class TestResult:
    """Result of testing a strategy on a specific period/interval."""

    strategy_id: str
    symbol: str
    interval: str
    start_date: str
    end_date: str
    result_type: str  # "training" or "test"
    return_pct: float
    max_drawdown_pct: float
    total_trades: int
    final_equity: float
    peak_equity: float
    created_at: datetime = field(default_factory=datetime.utcnow)

    def get_filename(self) -> str:
        """Generate filename for this result."""
        start = self.start_date.replace("-", "")
        end = self.end_date.replace("-", "")
        return f"{self.strategy_id}_{self.interval}_{start}_{end}.json"

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy_id": self.strategy_id,
            "symbol": self.symbol,
            "interval": self.interval,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "result_type": self.result_type,
            "return_pct": self.return_pct,
            "max_drawdown_pct": self.max_drawdown_pct,
            "total_trades": self.total_trades,
            "final_equity": self.final_equity,
            "peak_equity": self.peak_equity,
            "created_at": self.created_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TestResult":
        return cls(
            strategy_id=data["strategy_id"],
            symbol=data["symbol"],
            interval=data["interval"],
            start_date=data["start_date"],
            end_date=data["end_date"],
            result_type=data["result_type"],
            return_pct=data["return_pct"],
            max_drawdown_pct=data["max_drawdown_pct"],
            total_trades=data["total_trades"],
            final_equity=data["final_equity"],
            peak_equity=data["peak_equity"],
            created_at=datetime.fromisoformat(data["created_at"]),
        )

    def save(self, directory: str | Path = "results") -> Path:
        """Save result to JSON file."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        filepath = directory / self.get_filename()

        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

        return filepath

    @classmethod
    def load(cls, filepath: str | Path) -> "TestResult":
        """Load result from JSON file."""
        with open(filepath) as f:
            data = json.load(f)

        return cls.from_dict(data)

    def __repr__(self) -> str:
        return f"TestResult({self.strategy_id}, {self.interval}, {self.result_type}, return={self.return_pct:.1f}%)"
