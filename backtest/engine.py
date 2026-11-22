"""Backtest engine."""

from dataclasses import dataclass
from typing: Optional, Callable
import pandas as pd
import numpy as np

from config import get_config
from signals import SignalGenerator, TradingSignal


@dataclass
class BacktestResult:
    """Backtest result."""
    equity_curve: pd.Series
    trades: pd.DataFrame
    metrics: dict
    signals: TradingSignal
    config: dict


class BacktestEngine:
    """
    Vectorized backtest engine.
    """

    def __init__(self, config: Optional[dict] = None):
        self.config = config or {}
        cfg = get_config()

        self.initial_capital = self.config.get("initial_capital", cfg.backtest.initial_capital)
        self.commission = self.config.get("commission", cfg.backtest.commission_per_contract)
        self.slippage_ticks = self.config.get("slippage_ticks", cfg.backtest.slippage_ticks)
        self.contract_multiplier = self.config.get("contract_multiplier", cfg.backtest.contract_multiplier)
        self.max_positions = self.config.get("max_positions", cfg.backtest.max_positions)
        self.position_sizing = self.config.get("position_sizing", cfg.backtest.position_sizing)
        self.risk_per_trade = self.config.get("risk_per_trade", cfg.backtest.risk_per_trade_pct)

        self.tick_size = 0.25
        self.slippage = self.slippage_ticks * self.tick_size

    def run(
        self,
        nq_bars: pd.DataFrame,
        qqq_bars: pd.DataFrame,
        options_chain: pd.DataFrame,
        footprint: pd.DataFrame,
        signal_generator: Optional[SignalGenerator] = None
    ) -> BacktestResult:
        """
        Run backtest.

        Args:
            nq_bars: NQ OHLCV bars
            qqq_bars: QQQ OHLCV bars
            options_chain: Options chain data
            footprint: Footprint data
            signal_generator: Optional custom signal generator

        Returns:
            BacktestResult
        """
        if signal_generator is None:
            signal_generator = SignalGenerator(self.config)

        # Generate signals
        signals = signal_generator.generate(nq_bars, qqq_bars, options_chain, footprint)

        # Extract trades from signals
        trades = self._extract_trades(nq_bars, signals)

        # Calculate metrics
        metrics = self._calculate_metrics(signals.equity_curve, trades)

        return BacktestResult(
            equity_curve=signals.equity_curve,
            trades=trades,
            metrics=metrics,
            signals=signals,
            config=self.config
        )

    def _extract_trades(self, bars: pd.DataFrame, signals: TradingSignal) -> pd.DataFrame:
        """Extract individual trades from position series."""
        position = signals.position
        entry = signals.entry
        exit = signals.exit

        trades = []
        in_trade = False
        trade_side = 0
        entry_idx = 0
        entry_price = 0
        entry_time = None
        stop_price = 0
        target_price = 0
        signal_type = ""

        for i in range(len(position)):
            curr_pos = position.iloc[i]
            prev_pos = position.iloc[i-1] if i > 0 else 0

            # Entry
            if prev_pos == 0 and curr_pos != 0:
                in_trade = True
                trade_side = curr_pos
                entry_idx = i
                entry_time = bars["timestamp"].iloc[i]
                if curr_pos == 1:
                    entry_price = entry.long_price.iloc[i]
                    stop_price = entry.stop_loss.iloc[i]
                    target_price = entry.take_profit.iloc[i]
                    signal_type = entry.signal_type.iloc[i]
                else:
                    entry_price = entry.short_price.iloc[i]
                    stop_price = entry.stop_loss.iloc[i]
                    target_price = entry.take_profit.iloc[i]
                    signal_type = entry.signal_type.iloc[i]

            # Exit
            elif prev_pos != 0 and curr_pos == 0 and in_trade:
                exit_time = bars["timestamp"].iloc[i]
                exit_price = exit.exit_price.iloc[i]
                exit_reason = exit.exit_reason.iloc[i]

                if trade_side == 1:
                    pnl = (exit_price - entry_price) * self.contract_multiplier
                else:
                    pnl = (entry_price - exit_price) * self.contract_multiplier

                # Subtract commission (entry + exit)
                pnl -= 2 * self.commission

                # Subtract slippage
                pnl -= 2 * self.slippage * self.contract_multiplier

                duration = (exit_time - entry_time).total_seconds() / 60  # minutes

                trades.append({
                    "entry_time": entry_time,
                    "exit_time": exit_time,
                    "side": "long" if trade_side == 1 else "short",
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "stop_price": stop_price,
                    "target_price": target_price,
                    "pnl": pnl,
                    "pnl_points": pnl / self.contract_multiplier,
                    "duration_min": duration,
                    "signal_type": signal_type,
                    "exit_reason": exit_reason,
                    "bars_held": i - entry_idx
                })

                in_trade = False
                trade_side = 0

        return pd.DataFrame(trades)

    def _calculate_metrics(self, equity: pd.Series, trades: pd.DataFrame) -> dict:
        """Calculate backtest metrics."""
        if len(trades) == 0:
            return {
                "total_return": 0,
                "annualized_return": 0,
                "sharpe_ratio": 0,
                "sortino_ratio": 0,
                "max_drawdown": 0,
                "max_drawdown_pct": 0,
                "win_rate": 0,
                "profit_factor": 0,
                "avg_win": 0,
                "avg_loss": 0,
                "expectancy": 0,
                "total_trades": 0,
                "winning_trades": 0,
                "losing_trades": 0,
                "avg_duration_min": 0,
                "calmar_ratio": 0
            }

        # Returns
        returns = equity.pct_change().dropna()
        total_return = (equity.iloc[-1] - equity.iloc[0]) / equity.iloc[0]

        # Annualized return (assuming 252 trading days, 390 min/day)
        periods_per_year = 252 * 390
        n_periods = len(returns)
        annualized_return = (1 + total_return) ** (periods_per_year / n_periods) - 1

        # Sharpe ratio
        risk_free = 0.02 / periods_per_year  # Daily risk free
        excess_returns = returns - risk_free
        sharpe = np.sqrt(periods_per_year) * excess_returns.mean() / excess_returns.std() if excess_returns.std() > 0 else 0

        # Sortino ratio
        downside_returns = returns[returns < 0]
        sortino = np.sqrt(periods_per_year) * excess_returns.mean() / downside_returns.std() if len(downside_returns) > 0 and downside_returns.std() > 0 else 0

        # Max drawdown
        cummax = equity.cummax()
        drawdown = (equity - cummax) / cummax
        max_drawdown = drawdown.min()
        max_drawdown_pct = max_drawdown * 100

        # Calmar ratio
        calmar = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0

        # Trade stats
        wins = trades[trades["pnl"] > 0]
        losses = trades[trades["pnl"] <= 0]

        win_rate = len(wins) / len(trades) if len(trades) > 0 else 0
        profit_factor = wins["pnl"].sum() / abs(losses["pnl"].sum()) if len(losses) > 0 and losses["pnl"].sum() != 0 else np.inf
        avg_win = wins["pnl"].mean() if len(wins) > 0 else 0
        avg_loss = losses["pnl"].mean() if len(losses) > 0 else 0
        expectancy = (win_rate * avg_win) + ((1 - win_rate) * avg_loss) if len(trades) > 0 else 0

        avg_duration = trades["duration_min"].mean()

        return {
            "total_return": total_return,
            "total_return_pct": total_return * 100,
            "annualized_return": annualized_return,
            "annualized_return_pct": annualized_return * 100,
            "sharpe_ratio": sharpe,
            "sortino_ratio": sortino,
            "max_drawdown": max_drawdown,
            "max_drawdown_pct": max_drawdown_pct,
            "win_rate": win_rate,
            "win_rate_pct": win_rate * 100,
            "profit_factor": profit_factor,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "expectancy": expectancy,
            "total_trades": len(trades),
            "winning_trades": len(wins),
            "losing_trades": len(losses),
            "avg_duration_min": avg_duration,
            "calmar_ratio": calmar,
            "final_equity": equity.iloc[-1],
            "initial_equity": equity.iloc[0]
        }


def run_walkforward_backtest(
    engine: BacktestEngine,
    nq_bars: pd.DataFrame,
    qqq_bars: pd.DataFrame,
    options_chain: pd.DataFrame,
    footprint: pd.DataFrame,
    train_months: int = 6,
    test_months: int = 1,
    step_months: int = 1,
    min_trades: int = 30
) -> list:
    """
    Run walk-forward backtest.

    Returns list of BacktestResult for each test period.
    """
    results = []

    # Get date range
    start_date = nq_bars["timestamp"].min()
    end_date = nq_bars["timestamp"].max()

    current_start = start_date
    while current_start < end_date:
        train_end = current_start + pd.DateOffset(months=train_months)
        test_end = train_end + pd.DateOffset(months=test_months)

        if test_end > end_date:
            test_end = end_date

        # Filter data
        train_mask = (nq_bars["timestamp"] >= current_start) & (nq_bars["timestamp"] < train_end)
        test_mask = (nq_bars["timestamp"] >= train_end) & (nq_bars["timestamp"] < test_end)

        train_nq = nq_bars[train_mask]
        test_nq = nq_bars[test_mask]
        train_qqq = qqq_bars[(qqq_bars["timestamp"] >= current_start) & (qqq_bars["timestamp"] < train_end)]
        test_qqq = qqq_bars[(qqq_bars["timestamp"] >= train_end) & (qqq_bars["timestamp"] < test_end)]
        train_opt = options_chain[(options_chain["timestamp"] >= current_start) & (options_chain["timestamp"] < train_end)]
        test_opt = options_chain[(options_chain["timestamp"] >= train_end) & (options_chain["timestamp"] < test_end)]
        train_fp = footprint[(footprint["timestamp"] >= current_start) & (footprint["timestamp"] < train_end)]
        test_fp = footprint[(footprint["timestamp"] >= train_end) & (footprint["timestamp"] < test_end)]

        if len(test_nq) == 0:
            break

        # Train (could optimize parameters here)
        # For now just run on test
        result = engine.run(test_nq, test_qqq, test_opt, test_fp)
        result.config["train_period"] = (current_start, train_end)
        result.config["test_period"] = (train_end, test_end)
        results.append(result)

        current_start += pd.DateOffset(months=step_months)

    return results