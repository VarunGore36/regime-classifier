"""Data validation utilities."""

import pandas as pd
import numpy as np
from typing: Optional, List, Dict, Any
from dataclasses import dataclass
from datetime import time


@dataclass
class ValidationResult:
    """Result of data validation."""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    stats: Dict[str, Any]


def validate_bars(
    df: pd.DataFrame,
    required_columns: Optional[List[str]] = None,
    timestamp_col: str = "timestamp",
    price_cols: Optional[List[str]] = None,
    volume_col: str = "volume",
    check_gaps: bool = True,
    expected_freq: str = "1min",
    session_open: time = time(9, 30),
    session_close: time = time(16, 0),
    tz: str = "America/New_York"
) -> ValidationResult:
    """
    Validate OHLCV bar data.

    Args:
        df: DataFrame with bar data
        required_columns: Required column names
        timestamp_col: Timestamp column name
        price_cols: Price column names (open, high, low, close)
        volume_col: Volume column name
        check_gaps: Whether to check for missing bars
        expected_freq: Expected frequency
        session_open: RTH session open time
        session_close: RTH session close time
        tz: Timezone

    Returns:
        ValidationResult
    """
    errors = []
    warnings = []
    stats = {}

    # Default columns
    if required_columns is None:
        required_columns = [timestamp_col, "open", "high", "low", "close", volume_col]
    if price_cols is None:
        price_cols = ["open", "high", "low", "close"]

    # Check required columns
    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        errors.append(f"Missing columns: {missing}")

    if errors:
        return ValidationResult(False, errors, warnings, stats)

    # Check for nulls
    for col in required_columns:
        null_count = df[col].isnull().sum()
        if null_count > 0:
            errors.append(f"Column '{col}' has {null_count} null values")

    # Check timestamp
    if not pd.api.types.is_datetime64_any_dtype(df[timestamp_col]):
        errors.append(f"Column '{timestamp_col}' is not datetime")

    if df[timestamp_col].dt.tz is None:
        warnings.append(f"Column '{timestamp_col}' is timezone-naive")

    # Check OHLC consistency
    for idx, row in df.iterrows():
        if row["high"] < max(row["open"], row["close"]):
            errors.append(f"Row {idx}: high < max(open, close)")
        if row["low"] > min(row["open"], row["close"]):
            errors.append(f"Row {idx}: low > min(open, close)")
        if row["high"] < row["low"]:
            errors.append(f"Row {idx}: high < low")

    # Check for negative prices/volume
    for col in price_cols:
        if (df[col] <= 0).any():
            errors.append(f"Column '{col}' has non-positive values")

    if (df[volume_col] < 0).any():
        errors.append(f"Column '{volume_col}' has negative values")

    # Check for duplicates
    dup = df[timestamp_col].duplicated().sum()
    if dup > 0:
        warnings.append(f"Found {dup} duplicate timestamps")

    # Check sorting
    if not df[timestamp_col].is_monotonic_increasing:
        warnings.append("Timestamps not monotonically increasing")

    # Check gaps
    if check_gaps and len(df) > 1:
        df_sorted = df.sort_values(timestamp_col)
        diffs = df_sorted[timestamp_col].diff().dropna()
        expected_td = pd.Timedelta(expected_freq)
        gaps = diffs[diffs > expected_td * 1.5]  # Allow 50% tolerance
        if len(gaps) > 0:
            warnings.append(f"Found {len(gaps)} gaps > 1.5x expected frequency")
            stats["max_gap"] = gaps.max()
            stats["gap_count"] = len(gaps)

    # Session stats
    if df[timestamp_col].dt.tz is not None:
        df_tz = df[timestamp_col].dt.tz_convert(tz)
        is_rth = (df_tz.dt.time >= session_open) & (df_tz.dt.time < session_close)
        stats["rth_bars"] = is_rth.sum()
        stats["eth_bars"] = (~is_rth).sum()
        stats["total_bars"] = len(df)

    # Price stats
    for col in price_cols:
        stats[f"{col}_min"] = df[col].min()
        stats[f"{col}_max"] = df[col].max()
        stats[f"{col}_mean"] = df[col].mean()

    stats["volume_total"] = df[volume_col].sum()
    stats["volume_mean"] = df[volume_col].mean()

    return ValidationResult(
        is_valid=len(errors) == 0,
        errors=errors,
        warnings=warnings,
        stats=stats
    )


def validate_options_chain(
    df: pd.DataFrame,
    required_columns: Optional[List[str]] = None
) -> ValidationResult:
    """Validate options chain data."""
    errors = []
    warnings = []
    stats = {}

    if required_columns is None:
        required_columns = [
            "timestamp", "expiry", "strike", "right",
            "oi", "volume", "iv",
            "delta", "gamma", "theta", "vega",
            "underlying_price"
        ]

    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        errors.append(f"Missing columns: {missing}")

    if errors:
        return ValidationResult(False, errors, warnings, stats)

    # Check right column
    if "right" in df.columns:
        invalid = df[~df["right"].isin(["C", "P", "c", "p"])]
        if len(invalid) > 0:
            errors.append(f"Invalid 'right' values in {len(invalid)} rows")

    # Check expiry >= timestamp
    if "expiry" in df.columns and "timestamp" in df.columns:
        early_exp = df[df["expiry"] < df["timestamp"]]
        if len(early_exp) > 0:
            warnings.append(f"{len(early_exp)} rows have expiry before timestamp")

    # Check IV
    if "iv" in df.columns:
        if (df["iv"] <= 0).any():
            errors.append("IV has non-positive values")
        if (df["iv"] > 5).any():
            warnings.append("IV > 500% found (possible data error)")

    # Check Greeks
    for greek in ["delta", "gamma", "theta", "vega"]:
        if greek in df.columns:
            if df[greek].isnull().any():
                warnings.append(f"Greek '{greek}' has null values")

    # Stats
    stats["total_rows"] = len(df)
    stats["unique_timestamps"] = df["timestamp"].nunique() if "timestamp" in df.columns else 0
    stats["unique_expiries"] = df["expiry"].nunique() if "expiry" in df.columns else 0
    stats["unique_strikes"] = df["strike"].nunique() if "strike" in df.columns else 0

    return ValidationResult(
        is_valid=len(errors) == 0,
        errors=errors,
        warnings=warnings,
        stats=stats
    )


def validate_footprint(
    df: pd.DataFrame,
    required_columns: Optional[List[str]] = None
) -> ValidationResult:
    """Validate footprint/order flow data."""
    errors = []
    warnings = []
    stats = {}

    if required_columns is None:
        required_columns = ["timestamp", "price", "bid_volume", "ask_volume"]

    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        errors.append(f"Missing columns: {missing}")

    if errors:
        return ValidationResult(False, errors, warnings, stats)

    # Check volumes
    if (df["bid_volume"] < 0).any() or (df["ask_volume"] < 0).any():
        errors.append("Negative volume found")

    # Check price
    if (df["price"] <= 0).any():
        errors.append("Non-positive price found")

    # Stats
    stats["total_rows"] = len(df)
    stats["unique_timestamps"] = df["timestamp"].nunique() if "timestamp" in df.columns else 0
    stats["unique_prices"] = df["price"].nunique() if "price" in df.columns else 0
    stats["total_bid_volume"] = df["bid_volume"].sum()
    stats["total_ask_volume"] = df["ask_volume"].sum()

    return ValidationResult(
        is_valid=len(errors) == 0,
        errors=errors,
        warnings=warnings,
        stats=stats
    )


def validate_alignment(
    *dfs: pd.DataFrame,
    timestamp_col: str = "timestamp",
    max_diff: pd.Timedelta = pd.Timedelta("1min")
) -> ValidationResult:
    """Validate multiple DataFrames are aligned on timestamp."""
    errors = []
    warnings = []
    stats = {}

    if len(dfs) < 2:
        return ValidationResult(True, [], ["Need at least 2 DataFrames to check alignment"], {})

    # Get timestamp ranges
    ranges = []
    for i, df in enumerate(dfs):
        if timestamp_col not in df.columns:
            errors.append(f"DataFrame {i}: missing '{timestamp_col}' column")
            continue
        ranges.append((df[timestamp_col].min(), df[timestamp_col].max()))

    if len(ranges) < 2:
        return ValidationResult(False, errors, warnings, stats)

    # Check overlap
    max_start = max(r[0] for r in ranges)
    min_end = min(r[1] for r in ranges)

    if max_start > min_end:
        errors.append("DataFrames have no overlapping time range")
    else:
        overlap = min_end - max_start
        stats["overlap_start"] = max_start
        stats["overlap_end"] = min_end
        stats["overlap_duration"] = overlap

    # Check alignment at common timestamps
    common_ts = set(dfs[0][timestamp_col])
    for df in dfs[1:]:
        common_ts &= set(df[timestamp_col])

    stats["common_timestamps"] = len(common_ts)

    if len(common_ts) == 0:
        warnings.append("No common timestamps found")

    return ValidationResult(
        is_valid=len(errors) == 0,
        errors=errors,
        warnings=warnings,
        stats=stats
    )