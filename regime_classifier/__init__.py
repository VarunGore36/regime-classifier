"""Regime Classifier - Main package."""

__version__ = "0.1.0"

from config import get_config, load_config
from data import NQDataLoader, QQQDataLoader, OptionsChainLoader, FootprintLoader, SyntheticDataLoader
from features import (
    calculate_volume_profile, calculate_vwap, calculate_realized_volatility,
    calculate_iv_features, calculate_greeks_for_chain, calculate_conversion_ratio,
    calculate_cvd, detect_absorption, detect_exhaustion, detect_trapped_traders
)
from regime import CombinedRegimeClassifier, RVZScoreRegimeClassifier, GEXRegimeClassifier
from signals import SignalGenerator, POCReversionSignal, VWAPBandSignal, BreakFadeSignal
from backtest import BacktestEngine, run_walkforward_backtest, bootstrap_backtest, monte_carlo_simulation
from reports import create_tearsheet, generate_report
from utils import (
    ensure_timezone, resample_bars, fill_missing_bars,
    calculate_session_stats, align_dataframes,
    calculate_returns, annualize_volatility, sharpe_ratio,
    sortino_ratio, max_drawdown, calmar_ratio,
    profit_factor, expectancy, streaks,
    rolling_metrics, time_weighted_return,
    information_ratio, beta, alpha
)
from validation import (
    validate_bars, validate_options_chain, validate_footprint,
    validate_alignment, ValidationResult
)
from pipeline import RegimePipeline, PipelineResult, run_pipeline

__all__ = [
    # Config
    "get_config", "load_config",
    # Data
    "NQDataLoader", "QQQDataLoader", "OptionsChainLoader", "FootprintLoader", "SyntheticDataLoader",
    # Features
    "calculate_volume_profile", "calculate_vwap", "calculate_realized_volatility",
    "calculate_iv_features", "calculate_greeks_for_chain", "calculate_conversion_ratio",
    "calculate_cvd", "detect_absorption", "detect_exhaustion", "detect_trapped_traders",
    # Regime
    "CombinedRegimeClassifier", "RVZScoreRegimeClassifier", "GEXRegimeClassifier",
    # Signals
    "SignalGenerator", "POCReversionSignal", "VWAPBandSignal", "BreakFadeSignal",
    # Backtest
    "BacktestEngine", "run_walkforward_backtest", "bootstrap_backtest", "monte_carlo_simulation",
    # Reports
    "create_tearsheet", "generate_report",
    # Utils
    "ensure_timezone", "resample_bars", "fill_missing_bars",
    "calculate_session_stats", "align_dataframes",
    "calculate_returns", "annualize_volatility", "sharpe_ratio",
    "sortino_ratio", "max_drawdown", "calmar_ratio",
    "profit_factor", "expectancy", "streaks",
    "rolling_metrics", "time_weighted_return",
    "information_ratio", "beta", "alpha",
    # Validation
    "validate_bars", "validate_options_chain", "validate_footprint",
    "validate_alignment", "ValidationResult",
    # Pipeline
    "RegimePipeline", "PipelineResult", "run_pipeline",
]