"""Population class for the GA trading system."""

import json
import os
import random
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Any

import pandas as pd

from jarvis.client import get_binance_client
from jarvis.ga.individual import Individual
from jarvis.logging import logger
from jarvis.models import ActionType
from jarvis.utils import datetime_to_timestamp, dt_range, interval_to_timedelta


def _evaluate_batch(
    individual_dicts: list[dict[str, Any]],
    preloaded_data: list[tuple[datetime, dict[str, Any], str]],  # (dt, df_dict, price_str)
    base_asset: str,
    trade_asset: str,
    starting_amount: str,
    commission_ratio: str,
    investment_ratio: str,
) -> list[float]:
    """Evaluate a batch of individuals. Runs in separate process."""
    from jarvis.ga.individual import Individual
    from jarvis.models import ActionType

    starting_amt = Decimal(starting_amount)
    comm_ratio = Decimal(commission_ratio)
    inv_ratio = Decimal(investment_ratio)

    # Pre-convert DataFrames once for this batch
    dfs_and_prices = [(pd.DataFrame(df_dict), Decimal(price_str)) for dt, df_dict, price_str in preloaded_data]

    results = []
    for individual_dict in individual_dicts:
        individual = Individual.from_dict(individual_dict)
        assets: dict[str, Decimal] = {base_asset: starting_amt}
        last_price = Decimal(0)

        for df, price in dfs_and_prices:
            last_price = price
            signal = individual.get_signal(df)

            if signal == ActionType.BUY:
                quote_balance = assets.get(base_asset, Decimal(0))
                spend_amount = quote_balance * inv_ratio
                if spend_amount > 0 and last_price > 0:
                    after_fee = spend_amount * (1 - comm_ratio)
                    buy_qty = after_fee / last_price
                    assets[base_asset] = quote_balance - spend_amount
                    assets[trade_asset] = assets.get(trade_asset, Decimal(0)) + buy_qty

            elif signal == ActionType.SELL:
                sell_qty = assets.get(trade_asset, Decimal(0))
                if sell_qty > 0 and last_price > 0:
                    proceeds = sell_qty * last_price
                    after_fee = proceeds * (1 - comm_ratio)
                    assets[trade_asset] = Decimal(0)
                    assets[base_asset] = assets.get(base_asset, Decimal(0)) + after_fee

        total = assets.get(base_asset, Decimal(0))
        trade_balance = assets.get(trade_asset, Decimal(0))
        if trade_balance > 0 and last_price > 0:
            total += trade_balance * last_price
        results.append(float(total))

    return results


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
        individuals = [
            Individual.random(rules_per_individual, price_hint=price_hint) for _ in range(population_size)
        ]
        return cls(individuals=individuals, population_size=population_size)

    def evaluate_fitness(
        self,
        symbol: str,
        interval: str,
        start_dt: datetime,
        end_dt: datetime,
        starting_amount: Decimal = Decimal("100"),
        commission_ratio: Decimal = Decimal("0.001"),
        investment_ratio: Decimal = Decimal("0.2"),
    ) -> None:
        """Evaluate fitness for all individuals using backtest.

        Fitness = final balance after simulated trading.
        """
        base_asset = symbol[-4:] if symbol.endswith("USDT") else symbol[-3:]
        trade_asset = symbol[:-4] if symbol.endswith("USDT") else symbol[:-3]

        interval_as_timedelta = interval_to_timedelta(interval)

        # Pre-load all kline data once for the entire date range
        all_dts = list(dt_range(start_dt, end_dt, interval_as_timedelta))
        preloaded_data: list[tuple[datetime, pd.DataFrame, Decimal]] = []

        if all_dts:
            # Load data once using a temporary client
            temp_client = get_binance_client(
                fake=True,
                extra_params={"assets": {base_asset: starting_amount}, "commission_ratio": commission_ratio},
            )
            for dt in all_dts:
                df = self._get_klines_as_df(temp_client, symbol, interval, dt)
                if not df.empty and len(df) >= 50:
                    price = Decimal(str(df["close"].iloc[-1]))
                    preloaded_data.append((dt, df, price))

        # Use parallel processing only for large populations when not using TA-Lib
        # TA-Lib is so fast that serialization overhead outweighs parallelization benefit
        from jarvis.ga.indicators import USE_TALIB

        num_workers = min(os.cpu_count() or 4, len(self.individuals))
        use_parallel = os.environ.get("JARVIS_NO_PARALLEL") != "1" and not USE_TALIB

        if use_parallel and num_workers > 1 and len(self.individuals) >= 100:
            # Convert data for multiprocessing (must be picklable)
            serialized_data = [
                (dt, df.to_dict("list"), str(price)) for dt, df, price in preloaded_data
            ]

            # Split individuals into batches for each worker
            batch_size = len(self.individuals) // num_workers
            batches = []
            for i in range(num_workers):
                start_idx = i * batch_size
                end_idx = start_idx + batch_size if i < num_workers - 1 else len(self.individuals)
                batches.append([ind.to_dict() for ind in self.individuals[start_idx:end_idx]])

            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                futures = [
                    executor.submit(
                        _evaluate_batch,
                        batch,
                        serialized_data,
                        base_asset,
                        trade_asset,
                        str(starting_amount),
                        str(commission_ratio),
                        str(investment_ratio),
                    )
                    for batch in batches
                ]

                # Collect results and assign fitness
                idx = 0
                for future in futures:
                    batch_results = future.result()
                    for fitness in batch_results:
                        self.individuals[idx].fitness = fitness
                        idx += 1
        else:
            # Single-threaded fallback for small populations
            for individual in self.individuals:
                assets: dict[str, Decimal] = {base_asset: starting_amount}
                last_price = Decimal(0)

                for dt, df, price in preloaded_data:
                    last_price = price
                    signal = individual.get_signal(df)

                    if signal == ActionType.BUY:
                        quote_balance = assets.get(base_asset, Decimal(0))
                        spend_amount = quote_balance * investment_ratio
                        if spend_amount > 0 and last_price > 0:
                            after_fee = spend_amount * (1 - commission_ratio)
                            buy_qty = after_fee / last_price
                            assets[base_asset] = quote_balance - spend_amount
                            assets[trade_asset] = assets.get(trade_asset, Decimal(0)) + buy_qty

                    elif signal == ActionType.SELL:
                        sell_qty = assets.get(trade_asset, Decimal(0))
                        if sell_qty > 0 and last_price > 0:
                            proceeds = sell_qty * last_price
                            after_fee = proceeds * (1 - commission_ratio)
                            assets[trade_asset] = Decimal(0)
                            assets[base_asset] = assets.get(base_asset, Decimal(0)) + after_fee

                total = assets.get(base_asset, Decimal(0))
                trade_balance = assets.get(trade_asset, Decimal(0))
                if trade_balance > 0 and last_price > 0:
                    total += trade_balance * last_price
                individual.fitness = float(total)

    def _get_klines_as_df(
        self, client: Any, symbol: str, interval: str, dt: datetime, lookback: int = 100
    ) -> pd.DataFrame:
        """Get klines as pandas DataFrame for indicator calculation.

        Uses explicit endTime parameter to avoid freeze_time dependency.
        """
        try:
            end_ts = datetime_to_timestamp(dt)
            klines = client.get_klines(symbol=symbol, interval=interval, limit=lookback, endTime=end_ts)
            if not klines:
                return pd.DataFrame()

            # Klines are Kline objects, convert to dict then DataFrame
            df = pd.DataFrame([k.model_dump() for k in klines])

            # Convert Decimal columns to float for indicator calculations
            for col in ["open", "high", "low", "close", "volume"]:
                if col in df.columns:
                    df[col] = df[col].astype(float)

            return df

        except Exception as e:
            logger.debug("Failed to get klines: %s", e)
            return pd.DataFrame()

    def select_parents(self) -> tuple[Individual, Individual]:
        """Select two parents using tournament selection."""
        tournament_size = 3

        def tournament() -> Individual:
            contestants = random.sample(self.individuals, min(tournament_size, len(self.individuals)))
            return max(contestants, key=lambda x: x.fitness)

        return tournament(), tournament()

    def evolve(self) -> "Population":
        """Evolve to the next generation.

        1. Keep top individuals (elitism)
        2. Create children through crossover
        3. Apply mutation
        4. Add some random individuals to prevent overfitting
        """
        # Sort by fitness
        sorted_individuals = sorted(self.individuals, key=lambda x: x.fitness, reverse=True)

        # Elitism: keep top performers
        num_elite = max(1, int(self.population_size * self.elitism_ratio))
        new_individuals = sorted_individuals[:num_elite]

        # Add some random individuals (prevent overfitting)
        num_random = max(1, int(self.population_size * 0.1))
        for _ in range(num_random):
            new_individuals.append(Individual.random())

        # Fill rest with children
        while len(new_individuals) < self.population_size:
            parent1, parent2 = self.select_parents()
            child = Individual.crossover(parent1, parent2)
            child = child.mutate(self.mutation_rate)
            new_individuals.append(child)

        return Population(
            individuals=new_individuals[:self.population_size],
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
