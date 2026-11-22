"""NQ and QQQ data loaders."""

from datetime import datetime
from typing import Optional
import pandas as pd

from .base import DataLoader, FileDataLoader


class NQDataLoader(FileDataLoader):
    """NQ (Nasdaq 100 E-mini futures) data loader."""

    def __init__(
        self,
        data_path: str,
        timezone: str = "America/New_York",
        file_format: str = "parquet",
        continuous: bool = True
    ):
        super().__init__("NQ", data_path, timezone, file_format)
        self.continuous = continuous

    def load_bars(
        self,
        start: datetime,
        end: datetime,
        frequency: str = "1min"
    ) -> pd.DataFrame:
        df = self._load_file(self.data_path)
        df = self._standardize_bars(df)
        df = df[(df["timestamp"] >= pd.Timestamp(start, tz=self.timezone)) &
                (df["timestamp"] <= pd.Timestamp(end, tz=self.timezone))]

        if frequency != "1min":
            df = self._resample_bars(df, frequency)

        return df

    def load_ticks(
        self,
        start: datetime,
        end: datetime
    ) -> pd.DataFrame:
        # Tick data would be in separate files typically
        tick_path = self.data_path.replace(".parquet", "_ticks.parquet")
        df = self._load_file(tick_path)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        if df["timestamp"].dt.tz is None:
            df["timestamp"] = df["timestamp"].dt.tz_localize(self.timezone)
        else:
            df["timestamp"] = df["timestamp"].dt.tz_convert(self.timezone)
        df = df[(df["timestamp"] >= pd.Timestamp(start, tz=self.timezone)) &
                (df["timestamp"] <= pd.Timestamp(end, tz=self.timezone))]
        df["symbol"] = self.symbol
        return df.sort_values("timestamp").reset_index(drop=True)

    def _resample_bars(self, df: pd.DataFrame, frequency: str) -> pd.DataFrame:
        df = df.set_index("timestamp")
        resampled = df.resample(frequency).agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
            "symbol": "first"
        }).dropna().reset_index()
        return resampled


class QQQDataLoader(FileDataLoader):
    """QQQ (Nasdaq 100 ETF) data loader for options underlying."""

    def __init__(
        self,
        data_path: str,
        timezone: str = "America/New_York",
        file_format: str = "parquet"
    ):
        super().__init__("QQQ", data_path, timezone, file_format)

    def load_bars(
        self,
        start: datetime,
        end: datetime,
        frequency: str = "1min"
    ) -> pd.DataFrame:
        df = self._load_file(self.data_path)
        df = self._standardize_bars(df)
        df = df[(df["timestamp"] >= pd.Timestamp(start, tz=self.timezone)) &
                (df["timestamp"] <= pd.Timestamp(end, tz=self.timezone))]

        if frequency != "1min":
            df = self._resample_bars(df, frequency)

        return df

    def load_ticks(
        self,
        start: datetime,
        end: datetime
    ) -> pd.DataFrame:
        tick_path = self.data_path.replace(".parquet", "_ticks.parquet")
        df = self._load_file(tick_path)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        if df["timestamp"].dt.tz is None:
            df["timestamp"] = df["timestamp"].dt.tz_localize(self.timezone)
        else:
            df["timestamp"] = df["timestamp"].dt.tz_convert(self.timezone)
        df = df[(df["timestamp"] >= pd.Timestamp(start, tz=self.timezone)) &
                (df["timestamp"] <= pd.Timestamp(end, tz=self.timezone))]
        df["symbol"] = self.symbol
        return df.sort_values("timestamp").reset_index(drop=True)

    def _resample_bars(self, df: pd.DataFrame, frequency: str) -> pd.DataFrame:
        df = df.set_index("timestamp")
        resampled = df.resample(frequency).agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
            "symbol": "first"
        }).dropna().reset_index()
        return resampled


class SyntheticDataLoader(DataLoader):
    """Generate synthetic data for testing."""

    def __init__(
        self,
        symbol: str = "NQ",
        timezone: str = "America/New_York",
        start_price: float = 15000.0,
        volatility: float = 0.01,
        drift: float = 0.0
    ):
        super().__init__(symbol, timezone)
        self.start_price = start_price
        self.volatility = volatility
        self.drift = drift

    def load_bars(
        self,
        start: datetime,
        end: datetime,
        frequency: str = "1min"
    ) -> pd.DataFrame:
        import numpy as np

        freq_minutes = pd.Timedelta(frequency).total_seconds() / 60
        n_bars = int((pd.Timestamp(end) - pd.Timestamp(start)).total_seconds() / 60 / freq_minutes)

        np.random.seed(42)
        returns = np.random.normal(self.drift, self.volatility, n_bars)
        prices = self.start_price * np.exp(np.cumsum(returns))

        timestamps = pd.date_range(start, periods=n_bars, freq=frequency, tz=self.timezone)

        # Generate OHLC from close prices
        noise = np.random.normal(0, self.volatility * 0.5, (n_bars, 3))
        opens = prices * (1 + noise[:, 0])
        highs = np.maximum(opens, prices) * (1 + np.abs(noise[:, 1]))
        lows = np.minimum(opens, prices) * (1 - np.abs(noise[:, 2]))
        volumes = np.random.randint(100, 10000, n_bars)

        df = pd.DataFrame({
            "timestamp": timestamps,
            "open": opens,
            "high": highs,
            "low": lows,
            "close": prices,
            "volume": volumes,
            "symbol": self.symbol
        })

        return df

    def load_ticks(
        self,
        start: datetime,
        end: datetime
    ) -> pd.DataFrame:
        # Generate synthetic ticks from bars
        bars = self.load_bars(start, end, "1min")
        ticks = []
        for _, bar in bars.iterrows():
            n_ticks = max(1, bar["volume"] // 100)
            tick_times = pd.date_range(
                bar["timestamp"],
                periods=n_ticks + 1,
                freq="1s"
            )[:-1]
            prices = np.linspace(bar["open"], bar["close"], n_ticks)
            for i, (ts, price) in enumerate(zip(tick_times, prices)):
                ticks.append({
                    "timestamp": ts,
                    "price": price,
                    "size": np.random.randint(1, 10),
                    "side": np.random.choice(["buy", "sell"]),
                    "symbol": self.symbol
                })
        return pd.DataFrame(ticks)