"""End-to-end pipeline for regime classification and backtesting."""

from dataclasses import dataclass
from typing: Optional, Dict, Any
import pandas as pd
import numpy as np

from config import load_config, get_config
from data import get_provider, SyntheticDataLoader
from features import (
    calculate_volume_profile, calculate_vwap, calculate_realized_volatility,
    calculate_iv_features, calculate_greeks_for_chain, calculate_conversion_ratio,
    calculate_cvd, detect_absorption, detect_exhaustion, detect_trapped_traders,
    get_entropy_features, calculate_iv_skew, calculate_term_structure,
    calculate_garch_volatility, calculate_shadow_gamma
)
from regime import CombinedRegimeClassifier, CompositeAdvancedRegimeClassifier
from signals import SignalGenerator
from backtest import BacktestEngine, run_walkforward_backtest, bootstrap_backtest
from reports import generate_report, create_tearsheet
from utils import validate_bars, validate_options_chain, validate_footprint, validate_alignment


@dataclass
class PipelineResult:
    """Complete pipeline result."""
    regime_result: Any
    advanced_regime_result: Any
    signals: Any
    backtest_result: Any
    walkforward_results: list
    bootstrap_result: Any
    report_path: str


class RegimePipeline:
    """
    End-to-end pipeline for regime classification strategy.
    """

    def __init__(self, config_path: Optional[str] = None):
        self.config = load_config(config_path) if config_path else get_config()
        self.signal_generator = SignalGenerator(self.config.to_dict())
        self.backtest_engine = BacktestEngine(self.config.to_dict())
        self.regime_classifier = CombinedRegimeClassifier()
        self.advanced_regime_classifier = CompositeAdvancedRegimeClassifier()

    def run(
        self,
        nq_bars: pd.DataFrame,
        qqq_bars: pd.DataFrame,
        options_chain: pd.DataFrame,
        footprint: pd.DataFrame,
        output_dir: str = "reports",
        run_walkforward: bool = True,
        run_bootstrap: bool = True
    ) -> PipelineResult:
        """
        Run complete pipeline.

        Args:
            nq_bars: NQ OHLCV bars
            qqq_bars: QQQ OHLCV bars
            options_chain: Options chain data
            footprint: Footprint/order flow data
            output_dir: Output directory for reports
            run_walkforward: Whether to run walk-forward analysis
            run_bootstrap: Whether to run bootstrap validation

        Returns:
            PipelineResult
        """
        import os
        os.makedirs(output_dir, exist_ok=True)

        # 1. Validate data
        print("Validating data...")
        self._validate_data(nq_bars, qqq_bars, options_chain, footprint)

        # 2. Generate signals
        print("Generating signals...")
        signals = self.signal_generator.generate(nq_bars, qqq_bars, options_chain, footprint)

        # 3. Run backtest
        print("Running backtest...")
        backtest_result = self.backtest_engine.run(
            nq_bars, qqq_bars, options_chain, footprint, self.signal_generator
        )

        # 4. Print metrics
        self._print_metrics(backtest_result.metrics)

        # 5. Walk-forward analysis
        walkforward_results = []
        if run_walkforward:
            print("Running walk-forward analysis...")
            walkforward_results = run_walkforward_backtest(
                self.backtest_engine, nq_bars, qqq_bars, options_chain, footprint,
                train_months=6, test_months=1, step_months=1
            )
            self._print_walkforward_summary(walkforward_results)

        # 6. Bootstrap validation
        bootstrap_result = None
        if run_bootstrap:
            print("Running bootstrap validation...")
            bootstrap_result = bootstrap_backtest(
                self.backtest_engine, nq_bars, qqq_bars, options_chain, footprint,
                n_paths=1000, block_size=20, seed=42
            )
            self._print_bootstrap_summary(bootstrap_result)

        # 7. Generate report
        print("Generating report...")
        report_path = f"{output_dir}/backtest_report.pdf"
        generate_report(backtest_result, report_path)

        # 8. Regime classification
        regime_data = self._prepare_regime_data(nq_bars, options_chain)
        regime_result = self.regime_classifier.classify(regime_data)
        advanced_regime_result = self.advanced_regime_classifier.classify(regime_data)

        print("Pipeline complete!")
        return PipelineResult(
            regime_result=regime_result,
            advanced_regime_result=advanced_regime_result,
            signals=signals,
            backtest_result=backtest_result,
            walkforward_results=walkforward_results,
            bootstrap_result=bootstrap_result,
            report_path=report_path
        )

    def run_with_provider(
        self,
        provider_name: str,
        provider_kwargs: Dict[str, Any],
        start: pd.Timestamp,
        end: pd.Timestamp,
        output_dir: str = "reports",
        **kwargs
    ) -> PipelineResult:
        """
        Run pipeline with data provider.

        Args:
            provider_name: Name of provider (databento, ibkr, polygon, cboe, local)
            provider_kwargs: Provider-specific kwargs
            start: Start date
            end: End date
            output_dir: Output directory
            **kwargs: Additional arguments passed to run()
        """
        provider = get_provider(provider_name, **provider_kwargs)

        print(f"Loading data from {provider_name}...")
        nq_bars = provider.get_bars("NQ", start, end)
        qqq_bars = provider.get_bars("QQQ", start, end)
        options_chain = provider.get_options_chain("QQQ", start, end)
        footprint = provider.get_footprint("NQ", start, end)

        return self.run(nq_bars, qqq_bars, options_chain, footprint, output_dir, **kwargs)

    def run_synthetic(
        self,
        start: pd.Timestamp,
        end: pd.Timestamp,
        output_dir: str = "reports",
        **kwargs
    ) -> PipelineResult:
        """Run pipeline with synthetic data."""
        print("Generating synthetic data...")
        nq_loader = SyntheticDataLoader("NQ", start_price=15000, volatility=0.01, drift=0.0002)
        qqq_loader = SyntheticDataLoader("QQQ", start_price=350, volatility=0.01, drift=0.0002)

        nq_bars = nq_loader.load_bars(start, end, "1min")
        qqq_bars = qqq_loader.load_bars(start, end, "1min")
        options_chain = pd.DataFrame()
        footprint = pd.DataFrame()

        return self.run(nq_bars, qqq_bars, options_chain, footprint, output_dir, **kwargs)

    def _validate_data(
        self,
        nq_bars: pd.DataFrame,
        qqq_bars: pd.DataFrame,
        options_chain: pd.DataFrame,
        footprint: pd.DataFrame
    ):
        """Validate all input data."""
        nq_val = validate_bars(nq_bars)
        qqq_val = validate_bars(qqq_bars)

        if not nq_val.is_valid:
            raise ValueError(f"NQ data validation failed: {nq_val.errors}")
        if not qqq_val.is_valid:
            raise ValueError(f"QQQ data validation failed: {qqq_val.errors}")

        if len(options_chain) > 0:
            opt_val = validate_options_chain(options_chain)
            if not opt_val.is_valid:
                raise ValueError(f"Options data validation failed: {opt_val.errors}")

        if len(footprint) > 0:
            fp_val = validate_footprint(footprint)
            if not fp_val.is_valid:
                raise ValueError(f"Footprint data validation failed: {fp_val.errors}")

        # Check alignment
        if len(options_chain) > 0 and len(footprint) > 0:
            align_val = validate_alignment(nq_bars, qqq_bars, options_chain, footprint)
            if not align_val.is_valid:
                raise ValueError(f"Data alignment failed: {align_val.errors}")

        print("  Data validation passed")

    def _prepare_regime_data(self, nq_bars: pd.DataFrame, options_chain: pd.DataFrame) -> pd.DataFrame:
        """Prepare data for regime classification."""
        df = nq_bars[["timestamp", "close"]].copy()

        if len(options_chain) > 0:
            gex_by_time = options_chain.groupby("timestamp").apply(
                lambda x: (x["oi"] * x["gamma"] * 100 * x["underlying_price"]).sum()
            ).rename("gex_net")

            dex_by_time = options_chain.groupby("timestamp").apply(
                lambda x: (x["oi"] * x["delta"] * 100 * x["underlying_price"]).sum()
            ).rename("dex_net")

            gamma_flip = options_chain.groupby("timestamp").apply(
                lambda x: x.loc[(x["oi"] * x["gamma"] * 100 * x["underlying_price"]).cumsum().abs().idxmin(), "strike"]
            ).rename("gamma_flip")

            df = df.set_index("timestamp").join([gex_by_time, dex_by_time, gamma_flip]).reset_index()

        df["iv"] = np.nan
        df["rv"] = np.nan
        df["options_chain"] = [options_chain] * len(df) if len(options_chain) > 0 else None

        return df

    def _print_metrics(self, metrics: Dict):
        """Print backtest metrics."""
        print("\n=== BACKTEST RESULTS ===")
        key_metrics = [
            "total_return_pct", "annualized_return_pct", "sharpe_ratio",
            "sortino_ratio", "calmar_ratio", "max_drawdown_pct",
            "win_rate_pct", "profit_factor", "expectancy",
            "total_trades", "avg_duration_min"
        ]
        for k in key_metrics:
            if k in metrics:
                v = metrics[k]
                if isinstance(v, float):
                    print(f"  {k}: {v:.4f}")
                else:
                    print(f"  {k}: {v}")

    def _print_walkforward_summary(self, results: list):
        """Print walk-forward summary."""
        print("\n=== WALK-FORWARD RESULTS ===")
        for i, r in enumerate(results):
            print(f"  Period {i+1}: Return={r.metrics['total_return_pct']:.2f}%, "
                  f"Trades={r.metrics['total_trades']}, "
                  f"WinRate={r.metrics['win_rate_pct']:.1f}%, "
                  f"Sharpe={r.metrics['sharpe_ratio']:.2f}")

    def _print_bootstrap_summary(self, result):
        """Print bootstrap summary."""
        print("\n=== BOOTSTRAP RESULTS ===")
        print(f"  Profit Probability: {result.profit_probability:.2%}")
        print(f"  P-value: {result.p_value:.4f}")
        print(f"  Mean Return: {result.mean_return:.2f} pts")
        print(f"  Median Return: {result.percentile_50:.2f} pts")
        print(f"  95% CI: [{result.confidence_interval[0]:.2f}, {result.confidence_interval[1]:.2f}]")


def run_pipeline(
    config_path: Optional[str] = None,
    provider: str = "synthetic",
    start: str = "2024-01-01",
    end: str = "2024-12-31",
    output_dir: str = "reports",
    **provider_kwargs
) -> PipelineResult:
    """
    Convenience function to run pipeline.

    Args:
        config_path: Path to config YAML
        provider: Data provider (synthetic, databento, ibkr, polygon, cboe, local)
        start: Start date
        end: End date
        output_dir: Output directory
        **provider_kwargs: Provider-specific arguments
    """
    pipeline = RegimePipeline(config_path)
    start_ts = pd.Timestamp(start, tz="America/New_York")
    end_ts = pd.Timestamp(end, tz="America/New_York")

    if provider == "synthetic":
        return pipeline.run_synthetic(start_ts, end_ts, output_dir)
    else:
        return pipeline.run_with_provider(provider, provider_kwargs, start_ts, end_ts, output_dir)