"""Base classes for data loaders."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Optional
import pandas as pd


@dataclass
class BarData:
    """Standardized bar data."""
    timestamp: pd.Timestamp
    open: float
    high: float
    low: float
    close: float
    volume: int
    symbol: str


@dataclass
class TickData:
    """Standardized tick data."""
    timestamp: pd.Timestamp
    price: float
    size: int
    side: str  # 'buy' or 'sell'
    symbol: str


class DataLoader(ABC):
    """Abstract base class for data loaders."""

    def __init__(self, symbol: str, timezone: str = "America/New_York"):
        self.symbol = symbol
        self.timezone = timezone

    @abstractmethod
    def load_bars(
        self,
        start: datetime,
        end: datetime,
        frequency: str = "1min"
    ) -> pd.DataFrame:
        """Load OHLCV bars."""
        pass

    @abstractmethod
    def load_ticks(
        self,
        start: datetime,
        end: datetime
    ) -> pd.DataFrame:
        """Load tick data."""
        pass

    def _standardize_bars(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure bar DataFrame has required columns."""
        required = ["timestamp", "open", "high", "low", "close", "volume"]
        for col in required:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")

        df = df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        if df["timestamp"].dt.tz is None:
            df["timestamp"] = df["timestamp"].dt.tz_localize(self.timezone)
        else:
            df["timestamp"] = df["timestamp"].dt.tz_convert(self.timezone)
        df = df.sort_values("timestamp").reset_index(drop=True)
        df["symbol"] = self.symbol
        return df


class FileDataLoader(DataLoader):
    """Base class for file-based data loaders (parquet, csv)."""

    def __init__(
        self,
        symbol: str,
        data_path: str,
        timezone: str = "America/New_York",
        file_format: str = "parquet"
    ):
        super().__init__(symbol, timezone)
        self.data_path = data_path
        self.file_format = file_format

    def _load_file(self, path: str) -> pd.DataFrame:
        if self.file_format == "parquet":
            return pd.read_parquet(path)
        elif self.file_format == "csv":
            return pd.read_csv(path)
        else:
            raise ValueError(f"Unsupported format: {self.file_format}")