"""Data provider adapters for real data sources."""

from abc import ABC, abstractmethod
from typing: Optional, List, Dict, Any
import pandas as pd
import numpy as np
from datetime import datetime, time
import requests
import os


class DataProvider(ABC):
    """Abstract base class for data providers."""

    @abstractmethod
    def get_bars(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        frequency: str = "1min"
    ) -> pd.DataFrame:
        """Get OHLCV bars."""
        pass

    @abstractmethod
    def get_options_chain(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        expiries: Optional[List[datetime]] = None
    ) -> pd.DataFrame:
        """Get options chain."""
        pass

    @abstractmethod
    def get_footprint(
        self,
        symbol: str,
        start: datetime,
        end: datetime
    ) -> pd.DataFrame:
        """Get footprint/order flow data."""
        pass


class DatabentoProvider(DataProvider):
    """Databento data provider."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("DATABENTO_API_KEY")
        if not self.api_key:
            raise ValueError("DATABENTO_API_KEY required")

        self.base_url = "https://hist.databento.com/v1"
        self.session = requests.Session()
        self.session.headers.update({"Authorization": f"Bearer {self.api_key}"})

    def get_bars(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        frequency: str = "1min"
    ) -> pd.DataFrame:
        """Get OHLCV bars from Databento."""
        # Map frequency
        freq_map = {
            "1min": "1m",
            "5min": "5m",
            "15min": "15m",
            "1h": "1h",
            "1d": "1d"
        }
        db_freq = freq_map.get(frequency, "1m")

        params = {
            "dataset": "GLBX.MDP3",
            "symbols": symbol,
            "schema": "ohlcv-1m",
            "start": start.isoformat(),
            "end": end.isoformat(),
            "stype_in": "raw_symbol",
            "encoding": "csv"
        }

        # Databento uses different schema for different frequencies
        if frequency != "1min":
            params["schema"] = f"ohlcv-{db_freq}"

        response = self.session.get(f"{self.base_url}/timeseries.get_range", params=params)
        response.raise_for_status()

        df = pd.read_csv(pd.io.common.StringIO(response.text))
        df["timestamp"] = pd.to_datetime(df["ts_event"]).dt.tz_convert("America/New_York")
        df = df.rename(columns={
            "open": "open",
            "high": "high",
            "low": "low",
            "close": "close",
            "volume": "volume"
        })
        return df[["timestamp", "open", "high", "low", "close", "volume"]]

    def get_options_chain(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        expiries: Optional[List[datetime]] = None
    ) -> pd.DataFrame:
        """Get options chain from Databento."""
        # Options data would be from OPRA dataset
        params = {
            "dataset": "OPRA.PILLAR",
            "symbols": f"{symbol}*",  # All strikes
            "schema": "definition",  # or "ohlcv-1m" for quotes
            "start": start.isoformat(),
            "end": end.isoformat(),
            "encoding": "csv"
        }

        response = self.session.get(f"{self.base_url}/timeseries.get_range", params=params)
        response.raise_for_status()

        df = pd.read_csv(pd.io.common.StringIO(response.text))
        # Transform to standard format
        # This is simplified - real implementation would be more complex
        return df

    def get_footprint(
        self,
        symbol: str,
        start: datetime,
        end: datetime
    ) -> pd.DataFrame:
        """Get footprint from Databento (MBP-10 or trades)."""
        params = {
            "dataset": "GLBX.MDP3",
            "symbols": symbol,
            "schema": "mbp-10",  # Market by price
            "start": start.isoformat(),
            "end": end.isoformat(),
            "encoding": "csv"
        }

        response = self.session.get(f"{self.base_url}/timeseries.get_range", params=params)
        response.raise_for_status()

        df = pd.read_csv(pd.io.common.StringIO(response.text))
        # Transform MBP to footprint
        # Simplified
        return df


class IBKRProvider(DataProvider):
    """Interactive Brokers data provider (via ib_insync)."""

    def __init__(self, host: str = "127.0.0.1", port: int = 7497, client_id: int = 1):
        self.host = host
        self.port = port
        self.client_id = client_id
        self.ib = None

    def connect(self):
        """Connect to IBKR."""
        try:
            from ib_insync import IB
            self.ib = IB()
            self.ib.connect(self.host, self.port, self.client_id)
        except ImportError:
            raise ImportError("ib_insync required for IBKRProvider. Install with: pip install ib_insync")

    def disconnect(self):
        """Disconnect from IBKR."""
        if self.ib:
            self.ib.disconnect()

    def get_bars(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        frequency: str = "1min"
    ) -> pd.DataFrame:
        """Get bars from IBKR."""
        if not self.ib:
            self.connect()

        from ib_insync import Future, util

        contract = Future(symbol, exchange="CME")
        self.ib.qualifyContracts(contract)

        bars = self.ib.reqHistoricalData(
            contract,
            endDateTime=end.strftime("%Y%m%d %H:%M:%S"),
            durationStr=f"{(end - start).days} D",
            barSizeSetting=frequency.replace("min", " mins").replace("h", " hrs"),
            whatToShow="TRADES",
            useRTH=True,
            formatDate=1
        )

        df = util.df(bars)
        df["timestamp"] = pd.to_datetime(df["date"]).dt.tz_localize("America/New_York")
        return df[["timestamp", "open", "high", "low", "close", "volume"]]

    def get_options_chain(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        expiries: Optional[List[datetime]] = None
    ) -> pd.DataFrame:
        """Get options chain from IBKR."""
        if not self.ib:
            self.connect()

        from ib_insync import Option, util

        # Get option chain
        chains = self.ib.reqSecDefOptParams(symbol, "", "FUT", 0)
        # This is complex - simplified
        return pd.DataFrame()

    def get_footprint(
        self,
        symbol: str,
        start: datetime,
        end: datetime
    ) -> pd.DataFrame:
        """IBKR doesn't provide footprint directly."""
        return pd.DataFrame()


class PolygonProvider(DataProvider):
    """Polygon.io data provider."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("POLYGON_API_KEY")
        if not self.api_key:
            raise ValueError("POLYGON_API_KEY required")

        self.base_url = "https://api.polygon.io"
        self.session = requests.Session()

    def get_bars(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        frequency: str = "1min"
    ) -> pd.DataFrame:
        """Get bars from Polygon."""
        # Map symbol format
        poly_symbol = f"C:{symbol}" if not symbol.startswith("C:") else symbol

        params = {
            "apiKey": self.api_key,
            "timespan": "minute",
            "multiplier": 1,
            "from": start.strftime("%Y-%m-%d"),
            "to": end.strftime("%Y-%m-%d"),
            "adjusted": "true",
            "sort": "asc",
            "limit": 50000
        }

        url = f"{self.base_url}/v2/aggs/ticker/{poly_symbol}/range/1/minute"
        response = self.session.get(url, params=params)
        response.raise_for_status()

        data = response.json()
        if "results" not in data:
            return pd.DataFrame()

        df = pd.DataFrame(data["results"])
        df["timestamp"] = pd.to_datetime(df["t"], unit="ms").dt.tz_localize("UTC").dt.tz_convert("America/New_York")
        df = df.rename(columns={"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"})
        return df[["timestamp", "open", "high", "low", "close", "volume"]]

    def get_options_chain(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        expiries: Optional[List[datetime]] = None
    ) -> pd.DataFrame:
        """Get options from Polygon."""
        # Polygon has options endpoints
        # Simplified
        return pd.DataFrame()

    def get_footprint(
        self,
        symbol: str,
        start: datetime,
        end: datetime
    ) -> pd.DataFrame:
        """Polygon doesn't provide footprint."""
        return pd.DataFrame()


class CBOEProvider(DataProvider):
    """CBOE data provider for options."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("CBOE_API_KEY")
        self.base_url = "https://cdn.cboe.com/api/global"

    def get_bars(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        frequency: str = "1min"
    ) -> pd.DataFrame:
        """CBOE doesn't provide futures bars."""
        return pd.DataFrame()

    def get_options_chain(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        expiries: Optional[List[datetime]] = None
    ) -> pd.DataFrame:
        """Get options chain from CBOE."""
        # CBOE provides delayed options data
        # Would need to implement specific endpoints
        return pd.DataFrame()

    def get_footprint(
        self,
        symbol: str,
        start: datetime,
        end: datetime
    ) -> pd.DataFrame:
        """CBOE doesn't provide footprint."""
        return pd.DataFrame()


class LocalFileProvider(DataProvider):
    """Local file data provider (Parquet/CSV)."""

    def __init__(
        self,
        bars_path: str,
        options_path: Optional[str] = None,
        footprint_path: Optional[str] = None,
        file_format: str = "parquet"
    ):
        self.bars_path = bars_path
        self.options_path = options_path
        self.footprint_path = footprint_path
        self.file_format = file_format

    def _read_file(self, path: str) -> pd.DataFrame:
        if self.file_format == "parquet":
            return pd.read_parquet(path)
        elif self.file_format == "csv":
            return pd.read_csv(path)
        else:
            raise ValueError(f"Unsupported format: {self.file_format}")

    def get_bars(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        frequency: str = "1min"
    ) -> pd.DataFrame:
        """Get bars from local file."""
        df = self._read_file(self.bars_path)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        if df["timestamp"].dt.tz is None:
            df["timestamp"] = df["timestamp"].dt.tz_localize("America/New_York")
        else:
            df["timestamp"] = df["timestamp"].dt.tz_convert("America/New_York")

        mask = (df["timestamp"] >= pd.Timestamp(start, tz="America/New_York")) & \
               (df["timestamp"] <= pd.Timestamp(end, tz="America/New_York"))
        return df[mask].sort_values("timestamp").reset_index(drop=True)

    def get_options_chain(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        expiries: Optional[List[datetime]] = None
    ) -> pd.DataFrame:
        """Get options from local file."""
        if not self.options_path:
            return pd.DataFrame()

        df = self._read_file(self.options_path)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        if df["timestamp"].dt.tz is None:
            df["timestamp"] = df["timestamp"].dt.tz_localize("America/New_York")
        else:
            df["timestamp"] = df["timestamp"].dt.tz_convert("America/New_York")

        mask = (df["timestamp"] >= pd.Timestamp(start, tz="America/New_York")) & \
               (df["timestamp"] <= pd.Timestamp(end, tz="America/New_York"))
        return df[mask].sort_values("timestamp").reset_index(drop=True)

    def get_footprint(
        self,
        symbol: str,
        start: datetime,
        end: datetime
    ) -> pd.DataFrame:
        """Get footprint from local file."""
        if not self.footprint_path:
            return pd.DataFrame()

        df = self._read_file(self.footprint_path)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        if df["timestamp"].dt.tz is None:
            df["timestamp"] = df["timestamp"].dt.tz_localize("America/New_York")
        else:
            df["timestamp"] = df["timestamp"].dt.tz_convert("America/New_York")

        mask = (df["timestamp"] >= pd.Timestamp(start, tz="America/New_York")) & \
               (df["timestamp"] <= pd.Timestamp(end, tz="America/New_York"))
        return df[mask].sort_values("timestamp").reset_index(drop=True)


def get_provider(provider_name: str, **kwargs) -> DataProvider:
    """Factory function to get data provider."""
    providers = {
        "databento": DatabentoProvider,
        "ibkr": IBKRProvider,
        "polygon": PolygonProvider,
        "cboe": CBOEProvider,
        "local": LocalFileProvider,
    }

    if provider_name not in providers:
        raise ValueError(f"Unknown provider: {provider_name}. Available: {list(providers.keys())}")

    return providers[provider_name](**kwargs)