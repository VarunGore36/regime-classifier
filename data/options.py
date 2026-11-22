"""Options chain data loader."""

from datetime import datetime
from typing import List, Optional
import pandas as pd

from .base import FileDataLoader


class OptionsChainLoader(FileDataLoader):
    """
    Options chain data loader.

    Expected columns:
    - timestamp: datetime
    - expiry: datetime (expiration date)
    - strike: float
    - right: 'C' or 'P' (call/put)
    - oi: int (open interest)
    - volume: int
    - iv: float (implied volatility)
    - delta: float
    - gamma: float
    - theta: float
    - vega: float
    - charm: float (optional)
    - underlying_price: float (spot price at snapshot)
    """

    REQUIRED_COLUMNS = [
        "timestamp", "expiry", "strike", "right",
        "oi", "volume", "iv",
        "delta", "gamma", "theta", "vega",
        "underlying_price"
    ]

    def __init__(
        self,
        data_path: str,
        timezone: str = "America/New_York",
        file_format: str = "parquet"
    ):
        super().__init__("OPTIONS", data_path, timezone, file_format)

    def load_chain(
        self,
        start: datetime,
        end: datetime,
        expiries: Optional[List[datetime]] = None,
        strikes: Optional[List[float]] = None
    ) -> pd.DataFrame:
        """Load options chain for date range."""
        df = self._load_file(self.data_path)
        df = self._standardize_chain(df)

        df = df[(df["timestamp"] >= pd.Timestamp(start, tz=self.timezone)) &
                (df["timestamp"] <= pd.Timestamp(end, tz=self.timezone))]

        if expiries:
            expiry_ts = [pd.Timestamp(e, tz=self.timezone) for e in expiries]
            df = df[df["expiry"].isin(expiry_ts)]

        if strikes:
            df = df[df["strike"].isin(strikes)]

        return df.sort_values(["timestamp", "expiry", "strike", "right"]).reset_index(drop=True)

    def load_snapshot(
        self,
        timestamp: datetime,
        expiry: Optional[datetime] = None
    ) -> pd.DataFrame:
        """Load single options chain snapshot."""
        df = self._load_file(self.data_path)
        df = self._standardize_chain(df)

        ts = pd.Timestamp(timestamp, tz=self.timezone)
        df = df[df["timestamp"] == ts]

        if expiry:
            df = df[df["expiry"] == pd.Timestamp(expiry, tz=self.timezone)]

        return df.sort_values(["strike", "right"]).reset_index(drop=True)

    def _standardize_chain(self, df: pd.DataFrame) -> pd.DataFrame:
        for col in self.REQUIRED_COLUMNS:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")

        df = df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df["expiry"] = pd.to_datetime(df["expiry"])

        if df["timestamp"].dt.tz is None:
            df["timestamp"] = df["timestamp"].dt.tz_localize(self.timezone)
        else:
            df["timestamp"] = df["timestamp"].dt.tz_convert(self.timezone)

        if df["expiry"].dt.tz is None:
            df["expiry"] = df["expiry"].dt.tz_localize(self.timezone)
        else:
            df["expiry"] = df["expiry"].dt.tz_convert(self.timezone)

        df["dte"] = (df["expiry"] - df["timestamp"]).dt.total_seconds() / 86400
        df["right"] = df["right"].astype(str).str.upper()

        return df

    def get_gex_levels(self, snapshot: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate GEX (Gamma Exposure) by strike.
        GEX = OI * Gamma * 100 * Underlying_Price (per strike)
        """
        gex = snapshot.copy()
        gex["gex"] = gex["oi"] * gex["gamma"] * 100 * gex["underlying_price"]
        gex_by_strike = gex.groupby("strike").agg({
            "gex": "sum",
            "oi": "sum",
            "gamma": "mean"
        }).reset_index()
        gex_by_strike = gex_by_strike.sort_values("strike")
        return gex_by_strike

    def get_dex_levels(self, snapshot: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate DEX (Delta Exposure) by strike.
        DEX = OI * Delta * 100 * Underlying_Price
        """
        dex = snapshot.copy()
        dex["dex"] = dex["oi"] * dex["delta"] * 100 * dex["underlying_price"]
        dex_by_strike = dex.groupby("strike").agg({
            "dex": "sum",
            "oi": "sum",
            "delta": "mean"
        }).reset_index()
        dex_by_strike = dex_by_strike.sort_values("strike")
        return dex_by_strike

    def get_chex_levels(self, snapshot: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate CHEX (Charm Exposure) by strike.
        CHEX = OI * Charm * 100 * Underlying_Price
        """
        if "charm" not in snapshot.columns:
            raise ValueError("Charm column required for CHEX calculation")

        chex = snapshot.copy()
        chex["chex"] = chex["oi"] * chex["charm"] * 100 * chex["underlying_price"]
        chex_by_strike = chex.groupby("strike").agg({
            "chex": "sum",
            "oi": "sum",
            "charm": "mean"
        }).reset_index()
        chex_by_strike = chex_by_strike.sort_values("strike")
        return chex_by_strike

    def find_gamma_flip(self, snapshot: pd.DataFrame) -> float:
        """
        Find gamma flip strike (where net GEX changes sign).
        Positive GEX below flip, negative above (or vice versa).
        """
        gex_levels = self.get_gex_levels(snapshot)
        gex_levels = gex_levels.sort_values("strike")
        gex_levels["cum_gex"] = gex_levels["gex"].cumsum()

        # Find where cumulative GEX crosses zero
        cross_idx = (gex_levels["cum_gex"] * gex_levels["cum_gex"].shift(1) <= 0).idxmax()
        if cross_idx > 0:
            return gex_levels.loc[cross_idx, "strike"]
        return gex_levels["strike"].median()

    def find_call_put_walls(self, snapshot: pd.DataFrame) -> tuple:
        """
        Find call wall (max call GEX) and put wall (max put GEX).
        """
        calls = snapshot[snapshot["right"] == "C"]
        puts = snapshot[snapshot["right"] == "P"]

        call_gex = calls.groupby("strike")["gex"].sum() if "gex" in calls.columns else \
                   calls.groupby("strike").apply(lambda x: (x["oi"] * x["gamma"] * 100 * x["underlying_price"]).sum())
        put_gex = puts.groupby("strike")["gex"].sum() if "gex" in puts.columns else \
                  puts.groupby("strike").apply(lambda x: (x["oi"] * x["gamma"] * 100 * x["underlying_price"]).sum())

        call_wall = call_gex.idxmax() if len(call_gex) > 0 else None
        put_wall = put_gex.idxmax() if len(put_gex) > 0 else None

        return call_wall, put_wall

    def get_high_oi_strikes(self, snapshot: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
        """Get strikes with highest open interest."""
        oi_by_strike = snapshot.groupby(["strike", "right"])["oi"].sum().reset_index()
        oi_by_strike = oi_by_strike.sort_values("oi", ascending=False)
        return oi_by_strike.head(top_n)

    def get_high_gamma_strikes(self, snapshot: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
        """Get strikes with highest gamma concentration."""
        if "gex" not in snapshot.columns:
            snapshot = snapshot.copy()
            snapshot["gex"] = snapshot["oi"] * snapshot["gamma"] * 100 * snapshot["underlying_price"]

        gamma_by_strike = snapshot.groupby(["strike", "right"])["gex"].sum().reset_index()
        gamma_by_strike = gamma_by_strike.sort_values("gex", ascending=False)
        return gamma_by_strike.head(top_n)