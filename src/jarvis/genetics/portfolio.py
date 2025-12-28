"""Portfolio-based trading strategy with multiple coins.

A PortfolioIndividual contains strategies for multiple coins and is evaluated
based on the combined portfolio performance, not individual coin performance.
"""

import copy
import json
import random
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np

from jarvis.genetics.indicators import OHLCV
from jarvis.genetics.individual import Individual
from jarvis.logging import logger
from jarvis.models import ActionType, PositionSide

# Constants
LOOKBACK_PERIOD = 200
COOLDOWN_CANDLES = 24  # Candles to wait after stop-loss
MIN_POSITION_SIZE = 10.0  # Minimum position size in USD
DEFAULT_FEE_RATE = 0.0004  # 0.04% taker fee


@dataclass
class PortfolioIndividual:
    """A portfolio strategy containing rules for multiple coins.

    Unlike Individual which manages a single coin, PortfolioIndividual
    manages multiple coins simultaneously. Fitness is calculated based
    on the combined portfolio equity curve.
    """

    coin_strategies: dict[str, Individual] = field(default_factory=dict)
    fitness: float = 0.0

    # Portfolio-level settings
    max_allocation_per_coin: float = 0.20  # 20% max per coin
    stop_loss_pct: float = 20.0  # Close all if portfolio drops 20%

    def get_signal(self, symbol: str, ohlcv: OHLCV, current_side: PositionSide = PositionSide.NONE) -> ActionType:
        """Get trading signal for a specific coin.

        Args:
            symbol: Trading pair symbol (e.g., BTCUSDT)
            ohlcv: OHLCV data for the symbol
            current_side: Current position direction

        Returns:
            ActionType for the specified symbol
        """
        if symbol not in self.coin_strategies:
            return ActionType.STAY
        return self.coin_strategies[symbol].get_signal(ohlcv, current_side)

    def get_all_signals(
        self,
        ohlcv_data: dict[str, OHLCV],
        current_sides: dict[str, PositionSide],
    ) -> dict[str, ActionType]:
        """Get trading signals for all coins.

        Args:
            ohlcv_data: Dict mapping symbol to OHLCV data
            current_sides: Dict mapping symbol to current position side

        Returns:
            Dict mapping symbol to ActionType
        """
        signals = {}
        for symbol in self.coin_strategies:
            if symbol in ohlcv_data:
                current_side = current_sides.get(symbol, PositionSide.NONE)
                signals[symbol] = self.get_signal(symbol, ohlcv_data[symbol], current_side)
        return signals

    def get_score(self, symbol: str, ohlcv: OHLCV) -> float:
        """Get the raw score for a symbol (for debugging/display)."""
        if symbol not in self.coin_strategies:
            return 0.0
        return self.coin_strategies[symbol].get_total_score(ohlcv)

    def mutate(self, mutation_rate: float = 0.1, interval: str = "4h") -> "PortfolioIndividual":
        """Return a mutated copy of this portfolio.

        Each coin's strategy is mutated independently.
        """
        new_strategies = {}
        for symbol, individual in self.coin_strategies.items():
            new_strategies[symbol] = individual.mutate(mutation_rate, interval)
        return PortfolioIndividual(
            coin_strategies=new_strategies,
            fitness=0.0,
            max_allocation_per_coin=self.max_allocation_per_coin,
            stop_loss_pct=self.stop_loss_pct,
        )

    @classmethod
    def crossover(
        cls,
        parent1: "PortfolioIndividual",
        parent2: "PortfolioIndividual",
        interval: str = "4h",
    ) -> "PortfolioIndividual":
        """Create a child by combining strategies from two parents.

        For each coin, randomly choose the strategy from either parent.
        """
        child_strategies = {}
        all_symbols = set(parent1.coin_strategies.keys()) | set(parent2.coin_strategies.keys())

        for symbol in all_symbols:
            has_p1 = symbol in parent1.coin_strategies
            has_p2 = symbol in parent2.coin_strategies

            if has_p1 and has_p2:
                # Both parents have this coin - crossover the individuals
                child_strategies[symbol] = Individual.crossover(
                    parent1.coin_strategies[symbol],
                    parent2.coin_strategies[symbol],
                    interval,
                )
            elif has_p1:
                child_strategies[symbol] = copy.deepcopy(parent1.coin_strategies[symbol])
            else:
                child_strategies[symbol] = copy.deepcopy(parent2.coin_strategies[symbol])

        return cls(
            coin_strategies=child_strategies,
            fitness=0.0,
            max_allocation_per_coin=parent1.max_allocation_per_coin,
            stop_loss_pct=parent1.stop_loss_pct,
        )

    @classmethod
    def random(
        cls,
        symbols: list[str],
        rules_per_coin: int = 5,
        interval: str = "4h",
        max_allocation_per_coin: float = 0.20,
        stop_loss_pct: float = 20.0,
    ) -> "PortfolioIndividual":
        """Create a random portfolio with strategies for each symbol.

        Args:
            symbols: List of trading pair symbols
            rules_per_coin: Number of rules per coin strategy
            interval: Trading interval for period calculation
            max_allocation_per_coin: Maximum allocation per coin (0-1)
            stop_loss_pct: Portfolio stop-loss percentage
        """
        strategies = {}
        for symbol in symbols:
            strategies[symbol] = Individual.random(
                num_rules=rules_per_coin,
                interval=interval,
            )
        return cls(
            coin_strategies=strategies,
            fitness=0.0,
            max_allocation_per_coin=max_allocation_per_coin,
            stop_loss_pct=stop_loss_pct,
        )

    @property
    def symbols(self) -> list[str]:
        """Get list of symbols in this portfolio."""
        return list(self.coin_strategies.keys())

    @property
    def total_rules(self) -> int:
        """Get total number of rules across all coins."""
        return sum(len(ind.rules) for ind in self.coin_strategies.values())

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary for JSON export."""
        return {
            "coin_strategies": {symbol: ind.to_dict() for symbol, ind in self.coin_strategies.items()},
            "fitness": self.fitness,
            "max_allocation_per_coin": self.max_allocation_per_coin,
            "stop_loss_pct": self.stop_loss_pct,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PortfolioIndividual":
        """Deserialize from dictionary."""
        strategies = {symbol: Individual.from_dict(ind_data) for symbol, ind_data in data["coin_strategies"].items()}
        return cls(
            coin_strategies=strategies,
            fitness=data.get("fitness", 0.0),
            max_allocation_per_coin=data.get("max_allocation_per_coin", 0.20),
            stop_loss_pct=data.get("stop_loss_pct", 20.0),
        )

    def __repr__(self) -> str:
        symbols_str = ", ".join(self.symbols[:3])
        if len(self.symbols) > 3:
            symbols_str += f", +{len(self.symbols) - 3} more"
        return f"PortfolioIndividual(coins=[{symbols_str}], rules={self.total_rules}, fitness={self.fitness:.2f})"


@dataclass
class Position:
    """An open position for backtesting."""

    side: PositionSide
    entry_price: float
    quantity: float
    allocated_amount: float

    def calculate_pnl(self, current_price: float) -> float:
        """Calculate unrealized P&L at current price."""
        if self.side == PositionSide.LONG:
            return self.quantity * (current_price - self.entry_price)
        return self.quantity * (self.entry_price - current_price)

    def calculate_exit_value(self, current_price: float, fee_rate: float) -> float:
        """Calculate value returned when closing position (allocated + pnl - fees)."""
        pnl = self.calculate_pnl(current_price)
        fee = (self.allocated_amount + self.quantity * current_price) * fee_rate
        return self.allocated_amount + pnl - fee

    def unrealized_value(self, current_price: float) -> float:
        """Calculate current value of position (allocated + unrealized P&L)."""
        return self.allocated_amount + self.calculate_pnl(current_price)


@dataclass
class PortfolioPopulation:
    """A population of portfolio strategies that evolve together.

    Unlike Population which evaluates individuals on a single coin,
    PortfolioPopulation evaluates based on combined portfolio performance.
    """

    individuals: list[PortfolioIndividual] = field(default_factory=list)
    generation: int = 0
    population_size: int = 50
    elitism_ratio: float = 0.1
    mutation_rate: float = 0.1

    @classmethod
    def create_random(
        cls,
        symbols: list[str],
        population_size: int = 50,
        rules_per_coin: int = 5,
        interval: str = "4h",
        max_allocation_per_coin: float = 0.20,
        stop_loss_pct: float = 20.0,
        seed_portfolio: PortfolioIndividual | None = None,
    ) -> "PortfolioPopulation":
        """Create a population of random portfolio strategies.

        Args:
            symbols: List of trading pair symbols
            population_size: Number of portfolio strategies
            rules_per_coin: Number of rules per coin
            interval: Trading interval
            max_allocation_per_coin: Max allocation per coin (0-1)
            stop_loss_pct: Portfolio stop-loss percentage
            seed_portfolio: If provided, create mutations from this seed
        """
        individuals = []

        if seed_portfolio is not None:
            individuals.append(seed_portfolio)
            remaining = population_size - 1

            # Light mutations (30%)
            for _ in range(int(remaining * 0.3)):
                individuals.append(seed_portfolio.mutate(0.1, interval))

            # Medium mutations (40%)
            for _ in range(int(remaining * 0.4)):
                individuals.append(seed_portfolio.mutate(0.3, interval))

            # Aggressive mutations (30%)
            while len(individuals) < population_size:
                individuals.append(seed_portfolio.mutate(0.6, interval))
        else:
            for _ in range(population_size):
                individuals.append(
                    PortfolioIndividual.random(
                        symbols=symbols,
                        rules_per_coin=rules_per_coin,
                        interval=interval,
                        max_allocation_per_coin=max_allocation_per_coin,
                        stop_loss_pct=stop_loss_pct,
                    )
                )

        return cls(individuals=individuals, population_size=population_size)

    def evaluate_fitness(
        self,
        symbols: list[str],
        interval: str,
        start_dt: datetime,
        end_dt: datetime,
        starting_balance: float = 1000.0,
        fee_rate: float = DEFAULT_FEE_RATE,
        preloaded_data: dict[str, list[tuple[OHLCV, float, datetime]]] | None = None,
    ) -> dict[str, list[tuple[OHLCV, float, datetime]]]:
        """Evaluate fitness for all portfolio strategies.

        Fitness is calculated based on portfolio equity curve:
        fitness = total_return - max_drawdown

        Args:
            symbols: List of trading pair symbols
            interval: Kline interval
            start_dt: Backtest start datetime
            end_dt: Backtest end datetime
            starting_balance: Initial portfolio balance
            fee_rate: Trading fee rate
            preloaded_data: Cached OHLCV data from previous generation

        Returns:
            preloaded_data for reuse in next generation
        """
        from jarvis.client import get_binance_client
        from jarvis.utils import datetime_to_timestamp, interval_to_timedelta

        # Load data only if not provided
        if preloaded_data is None:
            preloaded_data = {}
            client = get_binance_client(fake=True)
            lookback = LOOKBACK_PERIOD
            lookback_delta = interval_to_timedelta(interval) * lookback
            fetch_start = start_dt - lookback_delta

            for symbol in symbols:
                klines = client.get_klines(
                    symbol=symbol,
                    interval=interval,
                    startTime=datetime_to_timestamp(fetch_start),
                    endTime=datetime_to_timestamp(end_dt),
                    limit=50000,
                )

                if not klines:
                    logger.warning(f"No data for {symbol}")
                    continue

                n = len(klines)
                open_arr = np.array([float(k.open) for k in klines])
                high_arr = np.array([float(k.high) for k in klines])
                low_arr = np.array([float(k.low) for k in klines])
                close_arr = np.array([float(k.close) for k in klines])
                volume_arr = np.array([float(k.volume) for k in klines])

                symbol_data = []
                for i in range(lookback, n):
                    ohlcv = OHLCV(
                        open=open_arr[i - lookback + 1 : i + 1],
                        high=high_arr[i - lookback + 1 : i + 1],
                        low=low_arr[i - lookback + 1 : i + 1],
                        close=close_arr[i - lookback + 1 : i + 1],
                        volume=volume_arr[i - lookback + 1 : i + 1],
                    )
                    price = close_arr[i]
                    time = klines[i].open_time
                    symbol_data.append((ohlcv, price, time))

                preloaded_data[symbol] = symbol_data
                logger.info(f"Loaded {len(symbol_data)} candles for {symbol}")

        # Find common length
        min_len = min(len(data) for data in preloaded_data.values())
        if min_len == 0:
            logger.error("No common data points across symbols")
            return preloaded_data

        # Evaluate each portfolio
        for portfolio in self.individuals:
            portfolio.fitness = self._evaluate_single_portfolio(
                portfolio=portfolio,
                preloaded_data=preloaded_data,
                min_len=min_len,
                starting_balance=starting_balance,
                fee_rate=fee_rate,
            )

        return preloaded_data

    def _evaluate_single_portfolio(
        self,
        portfolio: PortfolioIndividual,
        preloaded_data: dict[str, list[tuple[OHLCV, float, datetime]]],
        min_len: int,
        starting_balance: float,
        fee_rate: float,
    ) -> float:
        """Evaluate a single portfolio strategy.

        Returns fitness score = total_return_pct - max_drawdown_pct
        """
        balance = starting_balance
        peak_balance = starting_balance
        max_drawdown = 0.0
        positions: dict[str, Position] = {}

        allocation_pct = portfolio.max_allocation_per_coin
        stop_loss_pct = portfolio.stop_loss_pct
        stopped_out = False
        cooldown = 0

        for i in range(min_len):
            # Get current prices for all symbols
            prices = {}
            ohlcv_data = {}
            for symbol, data in preloaded_data.items():
                ohlcv, price, _ = data[i]
                prices[symbol] = price
                ohlcv_data[symbol] = ohlcv

            # Calculate portfolio value (balance + unrealized P&L)
            portfolio_value = balance + sum(
                pos.unrealized_value(prices.get(symbol, pos.entry_price)) for symbol, pos in positions.items()
            )

            # Track drawdown
            if portfolio_value > peak_balance:
                peak_balance = portfolio_value
            current_dd = (peak_balance - portfolio_value) / peak_balance * 100
            if current_dd > max_drawdown:
                max_drawdown = current_dd

            # Check portfolio stop-loss
            if current_dd >= stop_loss_pct and not stopped_out:
                # Close all positions at market
                for symbol, pos in list(positions.items()):
                    balance += pos.calculate_exit_value(prices[symbol], fee_rate)
                positions.clear()
                stopped_out = True
                cooldown = COOLDOWN_CANDLES
                continue

            # Cooldown after stop-loss
            if cooldown > 0:
                cooldown -= 1
                if cooldown == 0:
                    stopped_out = False
                    peak_balance = balance
                continue

            # Process signals for each coin
            for symbol in portfolio.symbols:
                if symbol not in prices:
                    continue

                price = prices[symbol]
                ohlcv = ohlcv_data[symbol]
                current_side = positions[symbol].side if symbol in positions else PositionSide.NONE

                signal = portfolio.get_signal(symbol, ohlcv, current_side)

                # Close position if needed
                if symbol in positions:
                    pos = positions[symbol]
                    should_close = (
                        signal == ActionType.CLOSE
                        or (signal == ActionType.LONG and pos.side == PositionSide.SHORT)
                        or (signal == ActionType.SHORT and pos.side == PositionSide.LONG)
                    )
                    if should_close:
                        balance += pos.calculate_exit_value(price, fee_rate)
                        del positions[symbol]

                # Open new position if needed
                if symbol not in positions and signal in (ActionType.LONG, ActionType.SHORT):
                    alloc = balance * allocation_pct
                    if alloc > MIN_POSITION_SIZE:
                        quantity = alloc / price
                        fee = alloc * fee_rate
                        balance -= alloc + fee

                        side = PositionSide.LONG if signal == ActionType.LONG else PositionSide.SHORT
                        positions[symbol] = Position(
                            side=side,
                            entry_price=price,
                            quantity=quantity,
                            allocated_amount=alloc,
                        )

        # Close remaining positions at end
        for symbol, pos in positions.items():
            price = prices.get(symbol, pos.entry_price)
            balance += pos.calculate_exit_value(price, fee_rate)

        # Calculate fitness
        total_return_pct = (balance - starting_balance) / starting_balance * 100

        # Fitness = return - drawdown (penalize high drawdown)
        fitness = total_return_pct - max_drawdown

        return fitness

    def select_parents(self) -> tuple[PortfolioIndividual, PortfolioIndividual]:
        """Select two parents using tournament selection."""
        tournament_size = 3

        def tournament() -> PortfolioIndividual:
            contestants = random.sample(self.individuals, min(tournament_size, len(self.individuals)))
            return max(contestants, key=lambda x: x.fitness)

        return tournament(), tournament()

    def evolve(self, interval: str = "4h") -> "PortfolioPopulation":
        """Evolve to the next generation."""
        sorted_individuals = sorted(self.individuals, key=lambda x: x.fitness, reverse=True)

        # Elitism: keep top performers
        num_elite = max(1, int(self.population_size * self.elitism_ratio))
        new_individuals = sorted_individuals[:num_elite]

        # Add random individuals (prevent overfitting)
        num_random = max(1, int(self.population_size * 0.1))
        symbols = sorted_individuals[0].symbols
        for _ in range(num_random):
            new_individuals.append(
                PortfolioIndividual.random(
                    symbols=symbols,
                    interval=interval,
                    max_allocation_per_coin=sorted_individuals[0].max_allocation_per_coin,
                    stop_loss_pct=sorted_individuals[0].stop_loss_pct,
                )
            )

        # Fill rest with children
        while len(new_individuals) < self.population_size:
            parent1, parent2 = self.select_parents()
            child = PortfolioIndividual.crossover(parent1, parent2, interval)
            child = child.mutate(self.mutation_rate, interval)
            new_individuals.append(child)

        return PortfolioPopulation(
            individuals=new_individuals[: self.population_size],
            generation=self.generation + 1,
            population_size=self.population_size,
            elitism_ratio=self.elitism_ratio,
            mutation_rate=self.mutation_rate,
        )

    def get_best(self) -> PortfolioIndividual:
        """Get the portfolio with highest fitness."""
        return max(self.individuals, key=lambda x: x.fitness)

    def get_average_fitness(self) -> float:
        """Get average fitness of the population."""
        if not self.individuals:
            return 0.0
        return sum(ind.fitness for ind in self.individuals) / len(self.individuals)

    def __repr__(self) -> str:
        best = self.get_best() if self.individuals else None
        avg = self.get_average_fitness()
        return f"PortfolioPopulation(gen={self.generation}, size={len(self.individuals)}, best={best.fitness if best else 0:.2f}, avg={avg:.2f})"


@dataclass
class PortfolioStrategy:
    """A saved portfolio strategy with metadata."""

    id: str
    symbols: list[str]
    created_at: datetime
    training_config: dict[str, Any]
    portfolio: PortfolioIndividual

    def save(self, filepath: str) -> None:
        """Save portfolio strategy to JSON file."""
        data = {
            "id": self.id,
            "symbols": self.symbols,
            "created_at": self.created_at.isoformat(),
            "training": self.training_config,
            "portfolio": self.portfolio.to_dict(),
        }
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, filepath: str) -> "PortfolioStrategy":
        """Load portfolio strategy from JSON file."""
        with open(filepath) as f:
            data = json.load(f)
        return cls(
            id=data["id"],
            symbols=data["symbols"],
            created_at=datetime.fromisoformat(data["created_at"]),
            training_config=data["training"],
            portfolio=PortfolioIndividual.from_dict(data["portfolio"]),
        )
