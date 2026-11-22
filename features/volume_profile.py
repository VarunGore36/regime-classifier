"""Volume Profile feature calculation."""

from dataclasses import dataclass
from typing import Optional, Tuple
import pandas as pd
import numpy as np

from config import get_config


@dataclass
class VolumeProfile:
    """Volume Profile results."""
    poc: float                          # Point of Control (price with max volume)
    vah: float                          # Value Area High (top 70% volume)
    val: float                          # Value Area Low (bottom 70% volume)
    hvns: pd.DataFrame                  # High Volume Nodes
    lvns: pd.DataFrame                  # Low Volume Nodes
    ledges: pd.DataFrame                # Ledges (below profile)
    profile: pd.DataFrame               # Full profile: price, volume, type
    value_area_volume: float            # Total volume in value area
    total_volume: float                 # Total session volume


def calculate_volume_profile(
    bars: pd.DataFrame,
    va_percentile: Optional[float] = None,
    hvn_threshold: Optional[float] = None,
    lvn_threshold: Optional[float] = None,
    tick_size: float = 0.25
) -> VolumeProfile:
    """
    Calculate Volume Profile from bars.

    Args:
        bars: DataFrame with columns [timestamp, open, high, low, close, volume]
        va_percentile: Value area percentile (default from config)
        hvn_threshold: HVN multiplier (default from config)
        lvn_threshold: LVN multiplier (default from config)
        tick_size: Minimum price increment

    Returns:
        VolumeProfile object
    """
    config = get_config()

    if va_percentile is None:
        va_percentile = config.volume_profile.va_percentile
    if hvn_threshold is None:
        hvn_threshold = config.volume_profile.hvn_threshold
    if lvn_threshold is None:
        lvn_threshold = config.volume_profile.lvn_threshold

    # Distribute volume across price levels within each bar
    price_levels = []

    for _, bar in bars.iterrows():
        # Create price range for this bar
        low_tick = int(bar["low"] / tick_size)
        high_tick = int(bar["high"] / tick_size)
        prices = np.arange(low_tick, high_tick + 1) * tick_size

        # Distribute volume proportionally (simplified: uniform)
        # More sophisticated: use TPO or volume-at-price from footprint
        vol_per_price = bar["volume"] / len(prices)

        for p in prices:
            price_levels.append({
                "price": p,
                "volume": vol_per_price,
                "timestamp": bar["timestamp"]
            })

    profile_df = pd.DataFrame(price_levels)
    profile_agg = profile_df.groupby("price")["volume"].sum().reset_index()
    profile_agg = profile_agg.sort_values("price").reset_index(drop=True)

    # Calculate POC
    poc_idx = profile_agg["volume"].idxmax()
    poc = profile_agg.loc[poc_idx, "price"]

    # Calculate Value Area (VA)
    total_volume = profile_agg["volume"].sum()
    target_volume = total_volume * va_percentile

    # Expand from POC outward until target volume reached
    poc_idx = profile_agg[profile_agg["price"] == poc].index[0]
    va_volume = profile_agg.loc[poc_idx, "volume"]
    lower_idx = poc_idx
    upper_idx = poc_idx

    while va_volume < target_volume and (lower_idx > 0 or upper_idx < len(profile_agg) - 1):
        # Check which side has more volume
        lower_vol = profile_agg.loc[lower_idx - 1, "volume"] if lower_idx > 0 else 0
        upper_vol = profile_agg.loc[upper_idx + 1, "volume"] if upper_idx < len(profile_agg) - 1 else 0

        if lower_vol >= upper_vol and lower_idx > 0:
            lower_idx -= 1
            va_volume += lower_vol
        elif upper_idx < len(profile_agg) - 1:
            upper_idx += 1
            va_volume += upper_vol
        else:
            break

    vah = profile_agg.loc[upper_idx, "price"]
    val = profile_agg.loc[lower_idx, "price"]

    # Identify HVNs (High Volume Nodes)
    avg_volume = profile_agg["volume"].mean()
    hvn_mask = profile_agg["volume"] >= (avg_volume * hvn_threshold)
    hvns = profile_agg[hvn_mask].copy()
    hvns["type"] = "HVN"

    # Identify LVNs (Low Volume Nodes)
    lvn_mask = profile_agg["volume"] <= (avg_volume * lvn_threshold)
    lvns = profile_agg[lvn_mask].copy()
    lvns["type"] = "LVN"

    # Identify Ledges (price levels below profile with volume)
    # Ledges are typically single prints at extremes
    ledge_mask = (profile_agg["volume"] > 0) & (profile_agg["volume"] < avg_volume * 0.5)
    ledges = profile_agg[ledge_mask].copy()
    ledges["type"] = "LEDGE"

    # Full profile with type labels
    full_profile = profile_agg.copy()
    full_profile["type"] = "NORMAL"
    full_profile.loc[full_profile["price"] == poc, "type"] = "POC"
    full_profile.loc[full_profile["price"].between(val, vah), "type"] = "VA"
    full_profile.loc[hvns.index, "type"] = "HVN"
    full_profile.loc[lvns.index, "type"] = "LVN"
    full_profile.loc[ledges.index, "type"] = "LEDGE"

    return VolumeProfile(
        poc=poc,
        vah=vah,
        val=val,
        hvns=hvns,
        lvns=lvns,
        ledges=ledges,
        profile=full_profile,
        value_area_volume=va_volume,
        total_volume=total_volume
    )


def calculate_tpo_profile(
    bars: pd.DataFrame,
    tick_size: float = 0.25
) -> VolumeProfile:
    """
    Calculate TPO (Time Price Opportunity) profile.
    Counts time spent at each price level instead of volume.
    """
    price_levels = []

    for _, bar in bars.iterrows():
        low_tick = int(bar["low"] / tick_size)
        high_tick = int(bar["high"] / tick_size)
        prices = np.arange(low_tick, high_tick + 1) * tick_size

        for p in prices:
            price_levels.append({
                "price": p,
                "tpo_count": 1,
                "timestamp": bar["timestamp"]
            })

    profile_df = pd.DataFrame(price_levels)
    profile_agg = profile_df.groupby("price")["tpo_count"].sum().reset_index()
    profile_agg = profile_agg.rename(columns={"tpo_count": "volume"})
    profile_agg = profile_agg.sort_values("price").reset_index(drop=True)

    # Reuse volume profile logic with TPO counts
    config = get_config()
    return calculate_volume_profile(
        bars.assign(volume=1),  # Dummy volume, we use TPO
        va_percentile=config.volume_profile.va_percentile,
        hvn_threshold=config.volume_profile.hvn_threshold,
        lvn_threshold=config.volume_profile.lvn_threshold,
        tick_size=tick_size
    )


def get_profile_levels(profile: VolumeProfile) -> dict:
    """Extract key levels from volume profile for signal generation."""
    return {
        "poc": profile.poc,
        "vah": profile.vah,
        "val": profile.val,
        "hvn_prices": profile.hvns["price"].tolist() if len(profile.hvns) > 0 else [],
        "lvn_prices": profile.lvns["price"].tolist() if len(profile.lvns) > 0 else [],
        "ledge_prices": profile.ledges["price"].tolist() if len(profile.ledges) > 0 else [],
    }