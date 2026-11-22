"""Command-line interface for regime classifier."""

import argparse
import sys
from pathlib import Path
import pandas as pd

from config import load_config, get_config
from data import SyntheticDataLoader
from backtest import BacktestEngine
from signals import SignalGenerator
from reports import generate_report


def create_synthetic_data(args):
    """Generate synthetic data for testing."""
    loader = SyntheticDataLoader(
        symbol=args.symbol,
        start_price=args.start_price,
        volatility=args.volatility,
        drift=args.drift
    )

    start = pd.Timestamp(args.start_date)
    end = pd.Timestamp(args.end_date)

    bars = loader.load_bars(start, end, args.frequency)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    bars.to_parquet(output_path, index=False)

    print(f"Generated {len(bars)} bars for {args.symbol}")
    print(f"Saved to {output_path}")


def run_backtest(args):
    """Run backtest with synthetic or real data."""
    config = load_config(args.config) if args.config else get_config()

    # Load data
    if args.synthetic:
        print("Using synthetic data...")
        nq_loader = SyntheticDataLoader("NQ", start_price=15000, volatility=0.01)
        qqq_loader = SyntheticDataLoader("QQQ", start_price=350, volatility=0.01)
        # Footprint and options would need synthetic generation too
        # For now, create minimal synthetic data
        start = pd.Timestamp(args.start_date)
        end = pd.Timestamp(args.end_date)

        nq_bars = nq_loader.load_bars(start, end, args.frequency)
        qqq_bars = qqq_loader.load_bars(start, end, args.frequency)

        # Create minimal synthetic options/footprint
        options_chain = pd.DataFrame()
        footprint = pd.DataFrame()
    else:
        print("Loading real data...")
        # Would load from files - implement based on your data format
        raise NotImplementedError("Real data loading not yet implemented. Use --synthetic for testing.")

    # Run backtest
    engine = BacktestEngine(config.to_dict())
    signal_generator = SignalGenerator(config.to_dict())

    print("Running backtest...")
    result = engine.run(nq_bars, qqq_bars, options_chain, footprint, signal_generator)

    # Print metrics
    print("\n" + "="*50)
    print("BACKTEST RESULTS")
    print("="*50)
    for key, value in result.metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")

    # Generate report
    if args.output:
        print(f"\nGenerating report to {args.output}...")
        generate_report(result, args.output, include_trades=True)
        print("Done!")

    return result


def run_walkforward(args):
    """Run walk-forward backtest."""
    config = load_config(args.config) if args.config else get_config()

    if args.synthetic:
        nq_loader = SyntheticDataLoader("NQ", start_price=15000, volatility=0.01)
        qqq_loader = SyntheticDataLoader("QQQ", start_price=350, volatility=0.01)

        start = pd.Timestamp(args.start_date)
        end = pd.Timestamp(args.end_date)

        nq_bars = nq_loader.load_bars(start, end, args.frequency)
        qqq_bars = qqq_loader.load_bars(start, end, args.frequency)
        options_chain = pd.DataFrame()
        footprint = pd.DataFrame()
    else:
        raise NotImplementedError("Real data loading not yet implemented.")

    from backtest import run_walkforward_backtest
    engine = BacktestEngine(config.to_dict())

    print("Running walk-forward backtest...")
    results = run_walkforward_backtest(
        engine, nq_bars, qqq_bars, options_chain, footprint,
        train_months=args.train_months,
        test_months=args.test_months,
        step_months=args.step_months
    )

    print(f"\nCompleted {len(results)} walk-forward periods")
    for i, r in enumerate(results):
        print(f"  Period {i+1}: Return={r.metrics['total_return_pct']:.2f}%, "
              f"Trades={r.metrics['total_trades']}, "
              f"WinRate={r.metrics['win_rate_pct']:.1f}%")

    if args.output:
        # Save summary
        summary = pd.DataFrame([{
            "period": i,
            "train_start": r.config.get("train_period", [None, None])[0],
            "train_end": r.config.get("train_period", [None, None])[1],
            "test_start": r.config.get("test_period", [None, None])[0],
            "test_end": r.config.get("test_period", [None, None])[1],
            "return_pct": r.metrics["total_return_pct"],
            "trades": r.metrics["total_trades"],
            "win_rate": r.metrics["win_rate_pct"],
            "sharpe": r.metrics["sharpe_ratio"],
            "max_dd": r.metrics["max_drawdown_pct"]
        } for i, r in enumerate(results)])
        summary.to_csv(args.output.replace('.pdf', '_wf.csv'), index=False)
        print(f"Walk-forward summary saved to {args.output.replace('.pdf', '_wf.csv')}")


def run_bootstrap(args):
    """Run bootstrap validation."""
    config = load_config(args.config) if args.config else get_config()

    if args.synthetic:
        nq_loader = SyntheticDataLoader("NQ", start_price=15000, volatility=0.01)
        qqq_loader = SyntheticDataLoader("QQQ", start_price=350, volatility=0.01)

        start = pd.Timestamp(args.start_date)
        end = pd.Timestamp(args.end_date)

        nq_bars = nq_loader.load_bars(start, end, args.frequency)
        qqq_bars = qqq_loader.load_bars(start, end, args.frequency)
        options_chain = pd.DataFrame()
        footprint = pd.DataFrame()
    else:
        raise NotImplementedError("Real data loading not yet implemented.")

    from backtest import bootstrap_backtest
    engine = BacktestEngine(config.to_dict())

    print("Running bootstrap validation...")
    result = bootstrap_backtest(
        engine, nq_bars, qqq_bars, options_chain, footprint,
        n_paths=args.n_paths,
        block_size=args.block_size,
        seed=args.seed
    )

    print("\n" + "="*50)
    print("BOOTSTRAP RESULTS")
    print("="*50)
    print(f"  Paths:              {result.n_paths}")
    print(f"  Profit Probability: {result.profit_probability:.2%}")
    print(f"  P-value:            {result.p_value:.4f}")
    print(f"  Mean Return:        {result.mean_return:.2f} pts")
    print(f"  Std Return:         {result.std_return:.2f} pts")
    print(f"  Median Return:      {result.percentile_50:.2f} pts")
    print(f"  5th Percentile:     {result.percentile_5:.2f} pts")
    print(f"  95th Percentile:    {result.percentile_95:.2f} pts")
    print(f"  CI ({args.confidence:.0%}):     [{result.confidence_interval[0]:.2f}, {result.confidence_interval[1]:.2f}]")


def main():
    parser = argparse.ArgumentParser(description="Regime Classifier - Futures/Options Strategy Backtester")
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Common arguments
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", type=str, help="Path to config YAML")
    common.add_argument("--synthetic", action="store_true", help="Use synthetic data")
    common.add_argument("--start-date", type=str, default="2024-01-01", help="Start date")
    common.add_argument("--end-date", type=str, default="2024-12-31", help="End date")
    common.add_argument("--frequency", type=str, default="1min", help="Bar frequency")
    common.add_argument("--output", type=str, help="Output file path")

    # Synthetic data command
    synth_parser = subparsers.add_parser("synthetic", parents=[common], help="Generate synthetic data")
    synth_parser.add_argument("--symbol", type=str, default="NQ", help="Symbol")
    synth_parser.add_argument("--start-price", type=float, default=15000, help="Start price")
    synth_parser.add_argument("--volatility", type=float, default=0.01, help="Daily volatility")
    synth_parser.add_argument("--drift", type=float, default=0.0, help="Daily drift")

    # Backtest command
    bt_parser = subparsers.add_parser("backtest", parents=[common], help="Run backtest")

    # Walk-forward command
    wf_parser = subparsers.add_parser("walkforward", parents=[common], help="Run walk-forward backtest")
    wf_parser.add_argument("--train-months", type=int, default=6, help="Training months")
    wf_parser.add_argument("--test-months", type=int, default=1, help="Test months")
    wf_parser.add_argument("--step-months", type=int, default=1, help="Step months")

    # Bootstrap command
    bs_parser = subparsers.add_parser("bootstrap", parents=[common], help="Run bootstrap validation")
    bs_parser.add_argument("--n-paths", type=int, default=1000, help="Number of bootstrap paths")
    bs_parser.add_argument("--block-size", type=int, default=20, help="Block size for stationary bootstrap")
    bs_parser.add_argument("--confidence", type=float, default=0.95, help="Confidence level")
    bs_parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    if args.command == "synthetic":
        create_synthetic_data(args)
    elif args.command == "backtest":
        run_backtest(args)
    elif args.command == "walkforward":
        run_walkforward(args)
    elif args.command == "bootstrap":
        run_bootstrap(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()