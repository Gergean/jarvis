"""Population class for the GA trading system."""

import json
import random
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Any

import numpy as np

from jarvis.client import get_binance_client
from jarvis.genetics.indicators import OHLCV
from jarvis.genetics.individual import Individual
from jarvis.logging import logger
from jarvis.models import (
    FUNDING_FEE_RATE,
    FUNDING_INTERVAL_HOURS,
    FUTURES_TAKER_FEE,
    ActionType,
    PositionSide,
)
from jarvis.utils import datetime_to_timestamp


@dataclass
class Population:
    """A population of trading strategies that can evolve over generations.

    Attributes:
        individuals: List of Individual trading strategies
        generation: Current generation number
        population_size: Target population size
        elitism_ratio: Fraction of best individuals to preserve each generation
        mutation_rate: Probability of mutating each rule
    """

    individuals: list[Individual] = field(default_factory=list)
    generation: int = 0
    population_size: int = 50
    elitism_ratio: float = 0.1
    mutation_rate: float = 0.1

    @classmethod
    def create_random(
        cls, population_size: int = 50, rules_per_individual: int = 5, price_hint: float | None = None
    ) -> "Population":
        """Create a random initial population.

        Args:
            population_size: Number of individuals
            rules_per_individual: Number of rules per individual
            price_hint: Approximate price of the asset for setting target ranges
        """
        individuals = [Individual.random(rules_per_individual, price_hint=price_hint) for _ in range(population_size)]
        return cls(individuals=individuals, population_size=population_size)

    def evaluate_fitness(
        self,
        symbol: str,
        interval: str,
        start_dt: datetime,
        end_dt: datetime,
        starting_margin: Decimal = Decimal("100"),
        commission_ratio: Decimal = FUTURES_TAKER_FEE,
        investment_ratio: Decimal = Decimal("0.2"),
        leverage: int = 1,
        funding_enabled: bool = True,
        preloaded_data: list[tuple[OHLCV, Decimal, int]] | None = None,
        buy_hold_return_pct: float | None = None,
    ) -> tuple[list[tuple[OHLCV, Decimal, int]], float]:
        """Evaluate fitness for all individuals using futures backtest.

        Fitness = strategy_return_pct - buy_hold_return_pct
        Positive fitness means the strategy beats buy & hold.

        Args:
            symbol: Trading pair symbol
            interval: Kline interval
            start_dt: Backtest start datetime
            end_dt: Backtest end datetime
            starting_margin: Initial USDT margin
            commission_ratio: Trading fee ratio (default: futures taker fee)
            investment_ratio: Fraction of margin to use per trade
            leverage: Futures leverage (1-10)
            funding_enabled: Whether to simulate funding fees
            preloaded_data: Cached OHLCV data from previous generation
            buy_hold_return_pct: Cached buy & hold return

        Returns:
            Tuple of (preloaded_data, buy_hold_return_pct) for reuse
        """
        # Load data only if not provided (first generation)
        if preloaded_data is None:
            preloaded_data = []

            temp_client = get_binance_client(
                fake=True,
                extra_params={"assets": {"USDT": starting_margin}, "commission_ratio": commission_ratio},
            )

            lookback = 200

            # Need to fetch extra data for lookback period
            from jarvis.utils import interval_to_timedelta
            lookback_delta = interval_to_timedelta(interval) * lookback
            fetch_start = start_dt - lookback_delta

            all_klines = temp_client.get_klines(
                symbol=symbol,
                interval=interval,
                startTime=datetime_to_timestamp(fetch_start),
                endTime=datetime_to_timestamp(end_dt),
                limit=50000,
            )

            first_price: Decimal | None = None
            last_price: Decimal | None = None

            if all_klines:
                n = len(all_klines)
                open_arr = np.zeros(n, dtype=np.float64)
                high_arr = np.zeros(n, dtype=np.float64)
                low_arr = np.zeros(n, dtype=np.float64)
                close_arr = np.zeros(n, dtype=np.float64)
                volume_arr = np.zeros(n, dtype=np.float64)

                for i, k in enumerate(all_klines):
                    open_arr[i] = float(k.open)
                    high_arr[i] = float(k.high)
                    low_arr[i] = float(k.low)
                    close_arr[i] = float(k.close)
                    volume_arr[i] = float(k.volume)

                for i in range(lookback, n):
                    start_idx = i - lookback + 1
                    end_idx = i + 1
                    ohlcv = OHLCV(
                        open=open_arr[start_idx:end_idx],
                        high=high_arr[start_idx:end_idx],
                        low=low_arr[start_idx:end_idx],
                        close=close_arr[start_idx:end_idx],
                        volume=volume_arr[start_idx:end_idx],
                    )
                    price = Decimal(str(close_arr[i]))
                    # Include candle index for funding fee calculation
                    preloaded_data.append((ohlcv, price, i))
                    if first_price is None:
                        first_price = price
                    last_price = price

            buy_hold_return_pct = 0.0
            if first_price and last_price and first_price > 0:
                buy_hold_return_pct = float((last_price - first_price) / first_price * 100)
                logger.info("Buy & Hold return: %.2f%% (%.4f -> %.4f)", buy_hold_return_pct, first_price, last_price)

            logger.info("Loaded %d evaluation points", len(preloaded_data))

        # Calculate funding interval in candles based on interval
        interval_hours = {"1m": 1/60, "5m": 5/60, "15m": 0.25, "30m": 0.5, "1h": 1, "4h": 4, "1d": 24}.get(interval, 1)
        funding_interval_candles = max(1, int(FUNDING_INTERVAL_HOURS / interval_hours))

        # Evaluate each individual
        for individual in self.individuals:
            margin_balance = starting_margin
            position_side = PositionSide.NONE
            position_entry_price = Decimal(0)
            position_quantity = Decimal(0)
            position_margin = Decimal(0)
            last_funding_candle = 0
            last_price = Decimal(0)

            for ohlcv, price, candle_idx in preloaded_data:
                last_price = price

                # Check liquidation if in position
                if position_side != PositionSide.NONE and leverage > 1:
                    liquidated = False
                    if position_side == PositionSide.LONG:
                        liq_price = position_entry_price * (1 - Decimal(1) / Decimal(leverage))
                        if price <= liq_price:
                            liquidated = True
                    else:  # SHORT
                        liq_price = position_entry_price * (1 + Decimal(1) / Decimal(leverage))
                        if price >= liq_price:
                            liquidated = True

                    if liquidated:
                        # Lose entire margin
                        margin_balance -= position_margin
                        position_side = PositionSide.NONE
                        position_quantity = Decimal(0)
                        position_margin = Decimal(0)
                        continue

                # Apply funding fee
                if funding_enabled and position_side != PositionSide.NONE:
                    candles_since_funding = candle_idx - last_funding_candle
                    if candles_since_funding >= funding_interval_candles:
                        num_funding_periods = candles_since_funding // funding_interval_candles
                        notional = position_quantity * price
                        funding_payment = notional * FUNDING_FEE_RATE * num_funding_periods
                        # Long pays, short receives (simplified)
                        if position_side == PositionSide.LONG:
                            margin_balance -= funding_payment
                        else:
                            margin_balance += funding_payment
                        last_funding_candle = candle_idx

                # Get signal with position awareness
                signal = individual.get_signal(ohlcv, position_side)

                if signal == ActionType.LONG and position_side == PositionSide.NONE:
                    # Open long position
                    margin_to_use = margin_balance * investment_ratio
                    if margin_to_use > 0 and price > 0:
                        position_size = (margin_to_use * leverage) / price
                        fee = position_size * price * commission_ratio
                        margin_balance -= fee
                        position_side = PositionSide.LONG
                        position_entry_price = price
                        position_quantity = position_size
                        position_margin = margin_to_use
                        margin_balance -= margin_to_use

                elif signal == ActionType.SHORT and position_side == PositionSide.NONE:
                    # Open short position
                    margin_to_use = margin_balance * investment_ratio
                    if margin_to_use > 0 and price > 0:
                        position_size = (margin_to_use * leverage) / price
                        fee = position_size * price * commission_ratio
                        margin_balance -= fee
                        position_side = PositionSide.SHORT
                        position_entry_price = price
                        position_quantity = position_size
                        position_margin = margin_to_use
                        margin_balance -= margin_to_use

                elif signal == ActionType.CLOSE and position_side != PositionSide.NONE:
                    # Close position and realize P&L
                    if position_side == PositionSide.LONG:
                        pnl = position_quantity * (price - position_entry_price)
                    else:  # SHORT
                        pnl = position_quantity * (position_entry_price - price)

                    fee = position_quantity * price * commission_ratio
                    margin_balance += position_margin + pnl - fee
                    position_side = PositionSide.NONE
                    position_quantity = Decimal(0)
                    position_margin = Decimal(0)

            # Calculate final equity (including unrealized P&L)
            final_equity = margin_balance
            if position_side != PositionSide.NONE and last_price > 0:
                if position_side == PositionSide.LONG:
                    unrealized_pnl = position_quantity * (last_price - position_entry_price)
                else:
                    unrealized_pnl = position_quantity * (position_entry_price - last_price)
                final_equity += position_margin + unrealized_pnl

            strategy_return_pct = float((final_equity - starting_margin) / starting_margin * 100)
            individual.fitness = strategy_return_pct - buy_hold_return_pct

        return preloaded_data, buy_hold_return_pct

    def select_parents(self) -> tuple[Individual, Individual]:
        """Select two parents using tournament selection."""
        tournament_size = 3

        def tournament() -> Individual:
            contestants = random.sample(self.individuals, min(tournament_size, len(self.individuals)))
            return max(contestants, key=lambda x: x.fitness)

        return tournament(), tournament()

    def evolve(self, price_hint: float | None = None) -> "Population":
        """Evolve to the next generation.

        1. Keep top individuals (elitism)
        2. Create children through crossover
        3. Apply mutation
        4. Add some random individuals to prevent overfitting

        Args:
            price_hint: Approximate price of the asset for random individuals
        """
        # Sort by fitness
        sorted_individuals = sorted(self.individuals, key=lambda x: x.fitness, reverse=True)

        # Elitism: keep top performers
        num_elite = max(1, int(self.population_size * self.elitism_ratio))
        new_individuals = sorted_individuals[:num_elite]

        # Add some random individuals (prevent overfitting)
        num_random = max(1, int(self.population_size * 0.1))
        for _ in range(num_random):
            new_individuals.append(Individual.random(price_hint=price_hint))

        # Fill rest with children
        while len(new_individuals) < self.population_size:
            parent1, parent2 = self.select_parents()
            child = Individual.crossover(parent1, parent2)
            child = child.mutate(self.mutation_rate)
            new_individuals.append(child)

        return Population(
            individuals=new_individuals[: self.population_size],
            generation=self.generation + 1,
            population_size=self.population_size,
            elitism_ratio=self.elitism_ratio,
            mutation_rate=self.mutation_rate,
        )

    def get_best(self) -> Individual:
        """Get the individual with highest fitness."""
        return max(self.individuals, key=lambda x: x.fitness)

    def get_average_fitness(self) -> float:
        """Get average fitness of the population."""
        if not self.individuals:
            return 0.0
        return sum(ind.fitness for ind in self.individuals) / len(self.individuals)

    def to_dict(self) -> dict[str, Any]:
        """Serialize population to dictionary."""
        return {
            "individuals": [ind.to_dict() for ind in self.individuals],
            "generation": self.generation,
            "population_size": self.population_size,
            "elitism_ratio": self.elitism_ratio,
            "mutation_rate": self.mutation_rate,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Population":
        """Deserialize population from dictionary."""
        individuals = [Individual.from_dict(ind) for ind in data["individuals"]]
        return cls(
            individuals=individuals,
            generation=data.get("generation", 0),
            population_size=data.get("population_size", 50),
            elitism_ratio=data.get("elitism_ratio", 0.1),
            mutation_rate=data.get("mutation_rate", 0.1),
        )

    def save(self, filepath: str) -> None:
        """Save population to JSON file."""
        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, filepath: str) -> "Population":
        """Load population from JSON file."""
        with open(filepath) as f:
            data = json.load(f)
        return cls.from_dict(data)

    def __repr__(self) -> str:
        best = self.get_best() if self.individuals else None
        avg = self.get_average_fitness()
        return f"Population(gen={self.generation}, size={len(self.individuals)}, best={best.fitness if best else 0:.2f}, avg={avg:.2f})"
