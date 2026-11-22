"""Utility functions."""

import pandas as pd
import numpy as np
from typing: Optional, List, Tuple
from datetime import time


def ensure_timezone(
    df: pd.DataFrame,
    column: str = "timestamp",
    tz: str = "America/New_York"
) -> pd.DataFrame:
    """Ensure timestamp column is timezone-aware."""
    df = df.copy()
    if df[column].dt.tz is None:
        df[column] = df[column].dt.tz_localize(tz)
    else:
        df[column] = df[column].dt.tz_convert(tz)
    return df


def resample_bars(
    bars: pd.DataFrame,
    frequency: str,
    timestamp_col: str = "timestamp"
) -> pd.DataFrame:
    """Resample OHLCV bars to different frequency."""
    df = bars.set_index(timestamp_col)
    resampled = df.resample(frequency).agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum"
    }).dropna().reset_index()
    return resampled


def fill_missing_bars(
    bars: pd.DataFrame,
    frequency: str = "1min",
    timestamp_col: str = "timestamp",
    method: str = "ffill"
) -> pd.DataFrame:
    """Fill missing bars in time series."""
    df = bars.set_index(timestamp_col)
    df = df.resample(frequency).asfreq()
    if method == "ffill":
        df = df.ffill()
    elif method == "interpolate":
        df = df.interpolate()
    return df.reset_index()


def calculate_session_stats(
    bars: pd.DataFrame,
    session_open: time = time(9, 30),
    session_close: time = time(16, 0),
    timestamp_col: str = "timestamp",
    tz: str = "America/New_York"
) -> pd.DataFrame:
    """Calculate daily session statistics."""
    df = bars.copy()
    df["date"] = df[timestamp_col].dt.tz_convert(tz).dt.date
    df["is_rth"] = (
        (df[timestamp_col].dt.tz_convert(tz).dt.time >= session_open) &
        (df[timestamp_col].dt.tz_convert(tz).dt.time < session_close)
    )

    daily = df.groupby("date").apply(lambda x: pd.Series({
        "rth_open": x[x["is_rth"]]["open"].iloc[0] if x["is_rth"].any() else np.nan,
        "rth_high": x[x["is_rth"]]["high"].max() if x["is_rth"].any() else np.nan,
        "rth_low": x[x["is_rth"]]["low"].min() if x["is_rth"].any() else np.nan,
        "rth_close": x[x["is_rth"]]["close"].iloc[-1] if x["is_rth"].any() else np.nan,
        "rth_volume": x[x["is_rth"]]["volume"].sum() if x["is_rth"].any() else 0,
        "eth_volume": x[~x["is_rth"]]["volume"].sum(),
        "total_volume": x["volume"].sum(),
        "trades": len(x)
    })).reset_index()

    return daily


def align_dataframes(
    *dfs: pd.DataFrame,
    on: str = "timestamp",
    method: str = "nearest",
    tolerance: str = "1min"
) -> List[pd.DataFrame]:
    """Align multiple DataFrames on timestamp."""
    if len(dfs) < 2:
        return list(dfs)

    # Use first as reference
    ref = dfs[0].set_index(on)
    aligned = [dfs[0]]

    for df in dfs[1:]:
        df_idx = df.set_index(on)
        # Reindex to reference
        df_aligned = df_idx.reindex(ref.index, method=method, tolerance=pd.Timedelta(tolerance))
        df_aligned = df_aligned.reset_index()
        aligned.append(df_aligned)

    return aligned


def calculate_returns(
    prices: pd.Series,
    method: str = "log",
    periods: int = 1
) -> pd.Series:
    """Calculate returns from price series."""
    if method == "log":
        return np.log(prices / prices.shift(periods))
    elif method == "simple":
        return prices / prices.shift(periods) - 1
    else:
        raise ValueError(f"Unknown method: {method}")


def annualize_volatility(
    volatility: float,
    periods_per_year: int = 252 * 390  # 1-min bars
) -> float:
    """Annualize volatility."""
    return volatility * np.sqrt(periods_per_year)


def sharpe_ratio(
    returns: pd.Series,
    risk_free: float = 0.0,
    periods_per_year: int = 252 * 390
) -> float:
    """Calculate Sharpe ratio."""
    excess = returns - risk_free / periods_per_year
    if excess.std() == 0:
        return 0.0
    return np.sqrt(periods_per_year) * excess.mean() / excess.std()


def sortino_ratio(
    returns: pd.Series,
    risk_free: float = 0.0,
    periods_per_year: int = 252 * 390
) -> float:
    """Calculate Sortino ratio."""
    excess = returns - risk_free / periods_per_year
    downside = excess[excess < 0]
    if len(downside) == 0 or downside.std() == 0:
        return 0.0
    return np.sqrt(periods_per_year) * excess.mean() / downside.std()


def max_drawdown(equity: pd.Series) -> Tuple[float, pd.Timestamp, pd.Timestamp]:
    """Calculate maximum drawdown and dates."""
    cummax = equity.cummax()
    dd = (equity - cummax) / cummax
    max_dd = dd.min()
    trough = dd.idxmin()
    peak = cummax.loc[:trough].idxmax()
    return max_dd, peak, trough


def calmar_ratio(
    equity: pd.Series,
    periods_per_year: int = 252 * 390
) -> float:
    """Calculate Calmar ratio."""
    returns = equity.pct_change().dropna()
    ann_return = (1 + returns.mean()) ** periods_per_year - 1
    mdd, _, _ = max_drawdown(equity)
    if mdd == 0:
        return 0.0
    return ann_return / abs(mdd)


def profit_factor(trades: pd.DataFrame, pnl_col: str = "pnl") -> float:
    """Calculate profit factor from trades."""
    wins = trades[trades[pnl_col] > 0][pnl_col].sum()
    losses = abs(trades[trades[pnl_col] <= 0][pnl_col].sum())
    if losses == 0:
        return np.inf
    return wins / losses


def expectancy(trades: pd.DataFrame, pnl_col: str = "pnl") -> float:
    """Calculate expectancy from trades."""
    wins = trades[trades[pnl_col] > 0]
    losses = trades[trades[pnl_col] <= 0]
    win_rate = len(wins) / len(trades) if len(trades) > 0 else 0
    avg_win = wins[pnl_col].mean() if len(wins) > 0 else 0
    avg_loss = losses[pnl_col].mean() if len(losses) > 0 else 0
    return win_rate * avg_win + (1 - win_rate) * avg_loss


def streaks(series: pd.Series) -> pd.DataFrame:
    """Calculate winning/losing streaks."""
    # Convert to binary
    binary = (series > 0).astype(int)
    # Group consecutive
    groups = (binary != binary.shift()).cumsum()
    streak_lengths = groups.groupby(groups).size()
    streak_values = binary.groupby(groups).first()

    return pd.DataFrame({
        "streak_length": streak_lengths,
        "is_win": streak_values == 1
    })


def rolling_metrics(
    equity: pd.Series,
    window: int = 252 * 20,  # ~1 month of 1-min bars
    risk_free: float = 0.0
) -> pd.DataFrame:
    """Calculate rolling performance metrics."""
    returns = equity.pct_change().dropna()

    rolling = pd.DataFrame(index=returns.index)
    rolling["sharpe"] = returns.rolling(window).apply(
        lambda x: sharpe_ratio(x, risk_free), raw=False
    )
    rolling["sortino"] = returns.rolling(window).apply(
        lambda x: sortino_ratio(x, risk_free), raw=False
    )
    rolling["volatility"] = returns.rolling(window).std() * np.sqrt(252 * 390)
    rolling["max_dd"] = equity.rolling(window).apply(
        lambda x: max_drawdown(x)[0], raw=False
    )

    return rolling


def time_weighted_return(
    equity: pd.Series,
    cash_flows: pd.Series = None
) -> float:
    """Calculate time-weighted return (TWR)."""
    if cash_flows is None:
        # No external cash flows
        return equity.iloc[-1] / equity.iloc[0] - 1

    # With cash flows - simplified
    # Would need proper implementation for production
    return equity.iloc[-1] / equity.iloc[0] - 1


def information_ratio(
    returns: pd.Series,
    benchmark_returns: pd.Series,
    periods_per_year: int = 252 * 390
) -> float:
    """Calculate Information Ratio vs benchmark."""
    active_returns = returns - benchmark_returns
    tracking_error = active_returns.std()
    if tracking_error == 0:
        return 0.0
    return np.sqrt(periods_per_year) * active_returns.mean() / tracking_error


def beta(
    returns: pd.Series,
    benchmark_returns: pd.Series
) -> float:
    """Calculate beta vs benchmark."""
    cov = np.cov(returns.dropna(), benchmark_returns.dropna())[0, 1]
    bench_var = np.var(benchmark_returns.dropna())
    if bench_var == 0:
        return 0.0
    return cov / bench_var


def alpha(
    returns: pd.Series,
    benchmark_returns: pd.Series,
    risk_free: float = 0.0,
    periods_per_year: int = 252 * 390
) -> float:
    """Calculate Jensen's Alpha."""
    b = beta(returns, benchmark_returns)
    ann_return = (1 + returns.mean()) ** periods_per_year - 1
    bench_ann = (1 + benchmark_returns.mean()) ** periods_per_year - 1
    rf_ann = (1 + risk_free) - 1
    return ann_return - (rf_ann + b * (bench_ann - rf_ann))