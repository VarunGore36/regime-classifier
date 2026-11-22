"""Bootstrap validation for backtest results."""

from dataclasses import dataclass
from typing: Optional
import pandas as pd
import numpy as np
from scipy import stats

from config import get_config
from .engine import BacktestEngine, BacktestResult


@dataclass
class BootstrapResult:
    """Bootstrap validation result."""
    n_paths: int
    profit_probability: float
    p_value: float
    percentile_5: float
    percentile_25: float
    percentile_50: float
    percentile_75: float
    percentile_95: float
    mean_return: float
    std_return: float
    paths_returns: np.ndarray
    paths_equity: np.ndarray
    confidence_interval: tuple


def bootstrap_backtest(
    engine: BacktestEngine,
    nq_bars: pd.DataFrame,
    qqq_bars: pd.DataFrame,
    options_chain: pd.DataFrame,
    footprint: pd.DataFrame,
    n_paths: int = 1000,
    block_size: int = 20,
    confidence: float = 0.95,
    seed: int = 42
) -> BootstrapResult:
    """
    Bootstrap validation using stationary bootstrap (Politis & Romano).

    Resamples blocks of trades to preserve autocorrelation structure.
    """
    np.random.seed(seed)

    # Run base backtest
    base_result = engine.run(nq_bars, qqq_bars, options_chain, footprint)
    base_trades = base_result.trades

    if len(base_trades) == 0:
        return BootstrapResult(
            n_paths=0,
            profit_probability=0,
            p_value=1,
            percentile_5=0,
            percentile_25=0,
            percentile_50=0,
            percentile_75=0,
            percentile_95=0,
            mean_return=0,
            std_return=0,
            paths_returns=np.array([]),
            paths_equity=np.array([]),
            confidence_interval=(0, 0)
        )

    # Extract trade returns
    trade_returns = base_trades["pnl_points"].values  # P&L in points
    n_trades = len(trade_returns)

    # Stationary bootstrap
    path_returns = []
    path_equity = []

    for _ in range(n_paths):
        # Generate random block lengths (geometric distribution)
        p = 1.0 / block_size
        bootstrap_returns = []

        while len(bootstrap_returns) < n_trades:
            block_len = np.random.geometric(p)
            start_idx = np.random.randint(0, n_trades)
            end_idx = min(start_idx + block_len, n_trades)
            bootstrap_returns.extend(trade_returns[start_idx:end_idx])

        bootstrap_returns = np.array(bootstrap_returns[:n_trades])

        # Calculate equity curve
        equity = np.cumsum(bootstrap_returns)
        total_return = equity[-1]

        path_returns.append(total_return)
        path_equity.append(equity)

    path_returns = np.array(path_returns)
    path_equity = np.array(path_equity)

    # Statistics
    profit_prob = (path_returns > 0).mean()
    mean_return = path_returns.mean()
    std_return = path_returns.std()

    # Percentiles
    p5 = np.percentile(path_returns, 5)
    p25 = np.percentile(path_returns, 25)
    p50 = np.percentile(path_returns, 50)
    p75 = np.percentile(path_returns, 75)
    p95 = np.percentile(path_returns, 95)

    # P-value (probability of getting base result or worse under null of zero mean)
    base_return = base_trades["pnl_points"].sum()
    p_value = (path_returns <= base_return).mean() if base_return > 0 else (path_returns >= base_return).mean()

    # Confidence interval
    alpha = 1 - confidence
    ci_lower = np.percentile(path_returns, alpha/2 * 100)
    ci_upper = np.percentile(path_returns, (1 - alpha/2) * 100)

    return BootstrapResult(
        n_paths=n_paths,
        profit_probability=profit_prob,
        p_value=p_value,
        percentile_5=p5,
        percentile_25=p25,
        percentile_50=p50,
        percentile_75=p75,
        percentile_95=p95,
        mean_return=mean_return,
        std_return=std_return,
        paths_returns=path_returns,
        paths_equity=path_equity,
        confidence_interval=(ci_lower, ci_upper)
    )


def monte_carlo_simulation(
    trades: pd.DataFrame,
    n_simulations: int = 10000,
    initial_capital: float = 100000,
    contract_multiplier: float = 20,
    seed: int = 42
) -> dict:
    """
    Monte Carlo simulation of equity curves by resampling trades with replacement.
    """
    np.random.seed(seed)

    if len(trades) == 0:
        return {}

    trade_pnls = trades["pnl"].values
    n_trades = len(trade_pnls)

    all_equity = np.zeros((n_simulations, n_trades + 1))
    all_equity[:, 0] = initial_capital

    for i in range(n_simulations):
        # Resample trades with replacement
        sampled = np.random.choice(trade_pnls, size=n_trades, replace=True)
        equity = np.cumsum(np.concatenate([[initial_capital], sampled]))
        all_equity[i] = equity

    # Statistics across simulations
    final_equity = all_equity[:, -1]
    max_dd = np.zeros(n_simulations)

    for i in range(n_simulations):
        eq = all_equity[i]
        peak = np.maximum.accumulate(eq)
        dd = (eq - peak) / peak
        max_dd[i] = dd.min()

    return {
        "median_final_equity": np.median(final_equity),
        "mean_final_equity": np.mean(final_equity),
        "std_final_equity": np.std(final_equity),
        "profit_probability": (final_equity > initial_capital).mean(),
        "percentile_5": np.percentile(final_equity, 5),
        "percentile_25": np.percentile(final_equity, 25),
        "percentile_50": np.percentile(final_equity, 50),
        "percentile_75": np.percentile(final_equity, 75),
        "percentile_95": np.percentile(final_equity, 95),
        "median_max_drawdown": np.median(max_dd),
        "worst_case_dd": np.percentile(max_dd, 5),
        "all_equity_curves": all_equity,
        "all_final_equity": final_equity
    }


def ablation_test(
    engine: BacktestEngine,
    nq_bars: pd.DataFrame,
    qqq_bars: pd.DataFrame,
    options_chain: pd.DataFrame,
    footprint: pd.DataFrame,
    components: dict
) -> dict:
    """
    Ablation test: remove components one at a time.

    Args:
        components: Dict of {component_name: config_modifier_function}
    """
    results = {}

    # Base run
    base_result = engine.run(nq_bars, qqq_bars, options_chain, footprint)
    results["base"] = {
        "total_return": base_result.metrics["total_return"],
        "total_trades": base_result.metrics["total_trades"],
        "win_rate": base_result.metrics["win_rate"],
        "profit_factor": base_result.metrics["profit_factor"],
        "max_drawdown": base_result.metrics["max_drawdown"]
    }

    # Test each component removal
    for name, modifier in components.items():
        # Create modified config
        modified_engine = BacktestEngine(modifier(engine.config))
        result = modified_engine.run(nq_bars, qqq_bars, options_chain, footprint)
        results[f"without_{name}"] = {
            "total_return": result.metrics["total_return"],
            "total_trades": result.metrics["total_trades"],
            "win_rate": result.metrics["win_rate"],
            "profit_factor": result.metrics["profit_factor"],
            "max_drawdown": result.metrics["max_drawdown"]
        }

    return results