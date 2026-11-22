"""Implied Volatility / Expected Move feature calculation."""

from dataclasses import dataclass
from typing import Optional
import pandas as pd
import numpy as np

from config import get_config


@dataclass
class IVResult:
    """Implied Volatility results."""
    iv: pd.Series                      # IV per snapshot
    expected_move: pd.Series           # Expected move (1 std dev)
    iv_percentile: pd.Series           # IV percentile rank
    iv_rank: pd.Series                 # IV rank (0-100)
    iv_vs_rv: pd.Series                # IV / RV ratio (vol risk premium)


def calculate_expected_move(
    price: float,
    iv: float,
    dte: float,
    confidence: float = 0.68
) -> float:
    """
    Calculate expected move from IV.

    Formula: Price * IV * sqrt(DTE / 365) * Z-score

    For 68% confidence (1 std dev), Z = 1.0
    For 95% confidence (2 std dev), Z = 1.96

    Args:
        price: Current underlying price
        iv: Implied volatility (annualized, e.g., 0.15 for 15%)
        dte: Days to expiration
        confidence: Confidence level (0.68 = 1 std dev)

    Returns:
        Expected move in price points
    """
    from scipy import stats
    z = stats.norm.ppf((1 + confidence) / 2)
    return price * iv * np.sqrt(dte / 365) * z


def calculate_iv_features(
    options_chain: pd.DataFrame,
    underlying_price: pd.Series,
    dte_column: str = "dte",
    iv_column: str = "iv",
    confidence: float = 0.68
) -> IVResult:
    """
    Calculate IV features from options chain.

    Args:
        options_chain: DataFrame with IV per strike/expiry per timestamp
        underlying_price: Series of underlying prices indexed by timestamp
        dte_column: Column name for days to expiration
        iv_column: Column name for implied volatility
        confidence: Confidence level for expected move

    Returns:
        IVResult
    """
    # Get ATM IV per timestamp (closest strike to underlying)
    atm_iv = []

    for ts in options_chain["timestamp"].unique():
        chain_ts = options_chain[options_chain["timestamp"] == ts]
        price = underlying_price.get(ts, chain_ts["underlying_price"].iloc[0])

        # Find ATM strike
        calls = chain_ts[chain_ts["right"] == "C"]
        puts = chain_ts[chain_ts["right"] == "P"]

        if len(calls) > 0:
            atm_strike = calls.loc[(calls["strike"] - price).abs().idxmin(), "strike"]
            atm_call_iv = calls[calls["strike"] == atm_strike][iv_column].mean()
        else:
            atm_call_iv = np.nan

        if len(puts) > 0:
            atm_strike = puts.loc[(puts["strike"] - price).abs().idxmin(), "strike"]
            atm_put_iv = puts[puts["strike"] == atm_strike][iv_column].mean()
        else:
            atm_put_iv = np.nan

        atm_iv_val = np.nanmean([atm_call_iv, atm_put_iv])
        dte = chain_ts[dte_column].iloc[0]

        atm_iv.append({
            "timestamp": ts,
            "iv": atm_iv_val,
            "dte": dte,
            "underlying_price": price
        })

    iv_df = pd.DataFrame(atm_iv).set_index("timestamp").sort_index()

    # Expected move
    iv_df["expected_move"] = iv_df.apply(
        lambda row: calculate_expected_move(row["underlying_price"], row["iv"], row["dte"], confidence),
        axis=1
    )

    # IV Percentile (rolling 252 days)
    iv_df["iv_percentile"] = iv_df["iv"].rolling(252 * 390, min_periods=100).rank(pct=True) * 100

    # IV Rank (current vs 52-week high/low)
    iv_52w_high = iv_df["iv"].rolling(252 * 390, min_periods=100).max()
    iv_52w_low = iv_df["iv"].rolling(252 * 390, min_periods=100).min()
    iv_df["iv_rank"] = (iv_df["iv"] - iv_52w_low) / (iv_52w_high - iv_52w_low) * 100

    return IVResult(
        iv=iv_df["iv"],
        expected_move=iv_df["expected_move"],
        iv_percentile=iv_df["iv_percentile"],
        iv_rank=iv_df["iv_rank"],
        iv_vs_rv=pd.Series(index=iv_df.index, dtype=float)  # Filled later with RV
    )


def calculate_iv_rv_ratio(
    iv_result: IVResult,
    rv_series: pd.Series
) -> pd.Series:
    """
    Calculate IV / RV ratio (Volatility Risk Premium).
    > 1: IV > RV (options expensive)
    < 1: IV < RV (options cheap)
    """
    # Align indices
    common_idx = iv_result.iv.index.intersection(rv_series.index)
    iv_aligned = iv_result.iv.loc[common_idx]
    rv_aligned = rv_series.loc[common_idx]
    return iv_aligned / rv_aligned


def get_iv_signals(
    iv_result: IVResult,
    rv_series: Optional[pd.Series] = None,
    iv_percentile_high: float = 80,
    iv_percentile_low: float = 20
) -> pd.DataFrame:
    """
    Generate IV-based signals.
    """
    df = pd.DataFrame({
        "iv": iv_result.iv,
        "expected_move": iv_result.expected_move,
        "iv_percentile": iv_result.iv_percentile,
        "iv_rank": iv_result.iv_rank
    })

    df["iv_high"] = df["iv_percentile"] > iv_percentile_high
    df["iv_low"] = df["iv_percentile"] < iv_percentile_low
    iv_rv_ratio = iv_result.iv_vs_rv
    if rv_series is not None:
        iv_rv_ratio = calculate_iv_rv_ratio(iv_result, rv_series)
        df = df.join(iv_rv_ratio.rename("iv_rv_ratio"))
        df["iv_expensive"] = df["iv_rv_ratio"] > 1.2
        df["iv_cheap"] = df["iv_rv_ratio"] < 0.8

    # IV expansion/contraction
    df["iv_expanding"] = df["iv"].diff() > 0
    df["iv_contracting"] = df["iv"].diff() < 0

    return df