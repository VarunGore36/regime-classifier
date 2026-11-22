"""IV Skew and Term Structure features."""

from dataclasses import dataclass
from typing: Optional
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit


@dataclass
class IVSkewResult:
    """IV Skew calculation results."""
    skew_25d: pd.Series           # 25-delta risk reversal
    skew_10d: pd.Series           # 10-delta risk reversal
    atm_iv: pd.Series             # ATM IV
    skew_slope: pd.Series         # Skew slope (put IV - call IV per delta)
    kurtosis: pd.Series           # 25d butterfly (kurtosis)
    term_structure: pd.DataFrame  # IV by DTE


@dataclass
class IVTermStructureResult:
    """IV Term Structure results."""
    iv_by_dte: pd.DataFrame       # IV for each expiry
    slope: pd.Series              # Term structure slope
    curvature: pd.Series          # Term structure curvature
    atm_iv_30d: pd.Series         # 30-day ATM IV (VIX-like)
    iv_cone: pd.DataFrame         # IV percentile cone


def calculate_iv_skew(
    options_chain: pd.DataFrame,
    underlying_price: pd.Series,
    delta_col: str = "delta",
    iv_col: str = "iv",
    strike_col: str = "strike",
    right_col: str = "right",
    expiry_col: str = "expiry",
    timestamp_col: str = "timestamp"
) -> IVSkewResult:
    """
    Calculate IV skew metrics from options chain.

    Risk reversal = Put IV - Call IV at same delta
    Butterfly = (Put IV + Call IV) / 2 - ATM IV
    """
    # For each timestamp, calculate skew
    results = []

    for ts in options_chain[timestamp_col].unique():
        chain = options_chain[options_chain[timestamp_col] == ts].copy()
        spot = underlying_price.get(ts, chain["underlying_price"].iloc[0])

        # Get expiries
        expiries = chain[expiry_col].unique()

        for exp in expiries:
            exp_chain = chain[chain[expiry_col] == exp].copy()
            if len(exp_chain) < 4:
                continue

            calls = exp_chain[exp_chain[right_col] == "C"].sort_values(delta_col)
            puts = exp_chain[exp_chain[right_col] == "P"].sort_values(delta_col)

            if len(calls) < 2 or len(puts) < 2:
                continue

            # ATM IV (closest to 0.5 delta)
            atm_call = calls.iloc[(calls[delta_col] - 0.5).abs().argsort()[:1]]
            atm_put = puts.iloc[(puts[delta_col] + 0.5).abs().argsort()[:1]]
            atm_iv_val = np.nanmean([atm_call[iv_col].values[0], atm_put[iv_col].values[0]])

            # 25-delta risk reversal
            call_25d = calls.iloc[(calls[delta_col] - 0.25).abs().argsort()[:1]]
            put_25d = puts.iloc[(puts[delta_col] + 0.25).abs().argsort()[:1]]
            skew_25d = put_25d[iv_col].values[0] - call_25d[iv_col].values[0]

            # 10-delta risk reversal
            call_10d = calls.iloc[(calls[delta_col] - 0.10).abs().argsort()[:1]]
            put_10d = puts.iloc[(puts[delta_col] + 0.10).abs().argsort()[:1]]
            skew_10d = put_10d[iv_col].values[0] - call_10d[iv_col].values[0]

            # 25-delta butterfly (kurtosis)
            fly_25d = (put_25d[iv_col].values[0] + call_25d[iv_col].values[0]) / 2 - atm_iv_val

            # Skew slope (linear fit of IV vs delta)
            all_options = exp_chain.sort_values(delta_col)
            if len(all_options) >= 5:
                deltas = all_options[delta_col].values
                ivs = all_options[iv_col].values
                # Fit quadratic: IV = a*delta^2 + b*delta + c
                try:
                    coeffs = np.polyfit(deltas, ivs, 2)
                    skew_slope = coeffs[1]  # Linear term
                    skew_curvature = coeffs[0]  # Quadratic term
                except:
                    skew_slope = np.nan
                    skew_curvature = np.nan
            else:
                skew_slope = np.nan
                skew_curvature = np.nan

            results.append({
                "timestamp": ts,
                "expiry": exp,
                "atm_iv": atm_iv_val,
                "skew_25d": skew_25d,
                "skew_10d": skew_10d,
                "skew_slope": skew_slope,
                "skew_curvature": skew_curvature,
                "kurtosis_25d": fly_25d
            })

    df = pd.DataFrame(results)
    if len(df) == 0:
        return IVSkewResult(
            skew_25d=pd.Series(dtype=float),
            skew_10d=pd.Series(dtype=float),
            atm_iv=pd.Series(dtype=float),
            skew_slope=pd.Series(dtype=float),
            kurtosis=pd.Series(dtype=float),
            term_structure=pd.DataFrame()
        )

    # Pivot to time series
    df = df.set_index(["timestamp", "expiry"])

    skew_25d = df["skew_25d"].unstack().ffill()
    skew_10d = df["skew_10d"].unstack().ffill()
    atm_iv = df["atm_iv"].unstack().ffill()
    skew_slope = df["skew_slope"].unstack().ffill()
    kurtosis = df["kurtosis_25d"].unstack().ffill()

    # Use nearest expiry as primary
    primary_expiry = skew_25d.columns[0] if len(skew_25d.columns) > 0 else None

    return IVSkewResult(
        skew_25d=skew_25d[primary_expiry] if primary_expiry else pd.Series(dtype=float),
        skew_10d=skew_10d[primary_expiry] if primary_expiry else pd.Series(dtype=float),
        atm_iv=atm_iv[primary_expiry] if primary_expiry else pd.Series(dtype=float),
        skew_slope=skew_slope[primary_expiry] if primary_expiry else pd.Series(dtype=float),
        kurtosis=kurtosis[primary_expiry] if primary_expiry else pd.Series(dtype=float),
        term_structure=df.reset_index()
    )


def calculate_term_structure(
    options_chain: pd.DataFrame,
    iv_col: str = "iv",
    delta_col: str = "delta",
    right_col: str = "right",
    expiry_col: str = "expiry",
    timestamp_col: str = "timestamp",
    dte_col: str = "dte"
) -> IVTermStructureResult:
    """
    Calculate IV term structure.

    Fits IV = a + b*sqrt(DTE) + c*DTE
    """
    results = []

    for ts in options_chain[timestamp_col].unique():
        chain = options_chain[options_chain[timestamp_col] == ts].copy()

        # ATM IV per expiry
        atm_ivs = []
        for exp in chain[expiry_col].unique():
            exp_chain = chain[chain[expiry_col] == exp]
            calls = exp_chain[exp_chain[right_col] == "C"]
            puts = exp_chain[exp_chain[right_col] == "P"]

            if len(calls) == 0 or len(puts) == 0:
                continue

            atm_call = calls.iloc[(calls[delta_col] - 0.5).abs().argsort()[:1]]
            atm_put = puts.iloc[(puts[delta_col] + 0.5).abs().argsort()[:1]]
            atm_iv = np.nanmean([atm_call[iv_col].values[0], atm_put[iv_col].values[0]])

            dte = exp_chain[dte_col].iloc[0]
            atm_ivs.append({"dte": dte, "iv": atm_iv, "expiry": exp})

        if len(atm_ivs) < 2:
            continue

        atm_df = pd.DataFrame(atm_ivs).sort_values("dte")

        # Fit term structure: IV = a + b*sqrt(DTE) + c*DTE
        try:
            X = np.column_stack([np.ones(len(atm_df)), np.sqrt(atm_df["dte"]), atm_df["dte"]])
            y = atm_df["iv"].values
            coeffs = np.linalg.lstsq(X, y, rcond=None)[0]
            a, b, c = coeffs
        except:
            a = b = c = np.nan

        # 30-day ATM IV (VIX-like)
        iv_30d = a + b * np.sqrt(30) + c * 30 if not np.isnan(a) else np.nan

        results.append({
            "timestamp": ts,
            "term_a": a,      # Level
            "term_b": b,      # Slope (sqrt)
            "term_c": c,      # Curvature
            "atm_iv_30d": iv_30d,
            "iv_by_dte": atm_df.set_index("dte")["iv"].to_dict()
        })

    df = pd.DataFrame(results).set_index("timestamp")

    # IV cone (percentiles over lookback)
    lookback = 252 * 390  # 1 year of 1-min bars
    iv_cone = pd.DataFrame(index=df.index)
    for pct in [5, 25, 50, 75, 95]:
        iv_cone[f"p{pct}"] = df["atm_iv_30d"].rolling(lookback, min_periods=100).quantile(pct/100)

    return IVTermStructureResult(
        iv_by_dte=df["iv_by_dte"],
        slope=df["term_b"],
        curvature=df["term_c"],
        atm_iv_30d=df["atm_iv_30d"],
        iv_cone=iv_cone
    )


def get_skew_signals(
    skew_result: IVSkewResult,
    term_result: IVTermStructureResult,
    skew_threshold: float = 0.02,
    kurtosis_threshold: float = 0.01
) -> pd.DataFrame:
    """
    Generate signals from IV skew and term structure.
    """
    df = pd.DataFrame(index=skew_result.skew_25d.index)

    df["skew_25d"] = skew_result.skew_25d
    df["skew_10d"] = skew_result.skew_10d
    df["atm_iv"] = skew_result.atm_iv
    df["kurtosis"] = skew_result.kurtosis
    df["term_slope"] = term_result.slope
    df["atm_iv_30d"] = term_result.atm_iv_30d

    # Skew signals
    df["skew_steep"] = df["skew_25d"] > skew_threshold  # Puts expensive
    df["skew_flat"] = df["skew_25d"] < -skew_threshold  # Calls expensive
    df["kurtosis_high"] = df["kurtosis"] > kurtosis_threshold  # Fat tails expected

    # Term structure signals
    df["term_steep"] = df["term_slope"] > 0  # Contango (IV rises with DTE)
    df["term_inverted"] = df["term_slope"] < 0  # Backwardation

    # Combined
    df["skew_bullish"] = df["skew_steep"] & df["kurtosis_high"]  # Put protection bought
    df["skew_bearish"] = df["skew_flat"] & df["kurtosis_high"]   # Call speculation

    return df