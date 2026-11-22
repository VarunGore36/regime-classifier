"""Tearsheet generation for backtest results."""

from dataclasses import dataclass
from typing import Optional
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
from scipy import stats

from config import get_config
from backtest import BacktestResult


@dataclass
class TearsheetConfig:
    """Tearsheet configuration."""
    figsize: tuple = (14, 20)
    style: str = "seaborn-v0_8-darkgrid"
    save_path: Optional[str] = None
    format: str = "pdf"


def create_tearsheet(
    result: BacktestResult,
    config: Optional[TearsheetConfig] = None
) -> plt.Figure:
    """
    Create comprehensive tearsheet.

    Args:
        result: BacktestResult
        config: TearsheetConfig

    Returns:
        matplotlib Figure
    """
    if config is None:
        config = TearsheetConfig()

    plt.style.use(config.style)
    fig = plt.figure(figsize=config.figsize)
    gs = gridspec.GridSpec(6, 2, figure=fig, hspace=0.3, wspace=0.25)

    equity = result.equity_curve
    trades = result.trades
    metrics = result.metrics

    # 1. Equity Curve
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(equity.index, equity.values, linewidth=0.8, color='steelblue')
    ax1.fill_between(equity.index, equity.values, alpha=0.1, color='steelblue')
    ax1.set_title(f'Equity Curve | Total Return: {metrics["total_return_pct"]:.2f}% | Sharpe: {metrics["sharpe_ratio"]:.2f}', fontsize=12)
    ax1.set_ylabel('Equity ($)')
    ax1.grid(True, alpha=0.3)

    # Add drawdown overlay
    ax1_dd = ax1.twinx()
    cummax = equity.cummax()
    drawdown = (equity - cummax) / cummax * 100
    ax1_dd.fill_between(equity.index, drawdown.values, 0, alpha=0.1, color='red')
    ax1_dd.set_ylabel('Drawdown (%)', color='red')
    ax1_dd.tick_params(axis='y', labelcolor='red')
    ax1_dd.set_ylim(drawdown.min() * 1.5, 5)

    # 2. Drawdown
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.fill_between(equity.index, drawdown.values, 0, alpha=0.5, color='red')
    ax2.set_title(f'Drawdown | Max DD: {metrics["max_drawdown_pct"]:.2f}%', fontsize=11)
    ax2.set_ylabel('Drawdown (%)')
    ax2.grid(True, alpha=0.3)

    # 3. Underwater plot
    ax3 = fig.add_subplot(gs[1, 1])
    underwater = (equity / cummax - 1) * 100
    ax3.fill_between(equity.index, underwater.values, 0, alpha=0.5, color='darkred')
    ax3.set_title('Underwater Plot', fontsize=11)
    ax3.set_ylabel('% from Peak')
    ax3.grid(True, alpha=0.3)

    # 4. Returns Distribution
    ax4 = fig.add_subplot(gs[2, 0])
    returns = equity.pct_change().dropna() * 100
    ax4.hist(returns, bins=50, alpha=0.7, color='steelblue', edgecolor='white', density=True)
    # Normal overlay
    x = np.linspace(returns.min(), returns.max(), 100)
    ax4.plot(x, stats.norm.pdf(x, returns.mean(), returns.std()), 'r-', lw=1, label='Normal')
    ax4.axvline(0, color='black', linestyle='--', alpha=0.5)
    ax4.set_title('Returns Distribution', fontsize=11)
    ax4.set_xlabel('Return (%)')
    ax4.set_ylabel('Density')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # 5. QQ Plot
    ax5 = fig.add_subplot(gs[2, 1])
    stats.probplot(returns, dist="norm", plot=ax5)
    ax5.set_title('Q-Q Plot (Normal)', fontsize=11)
    ax5.grid(True, alpha=0.3)

    # 6. Monthly Returns Heatmap
    ax6 = fig.add_subplot(gs[3, :])
    monthly = equity.resample('M').last().pct_change().dropna() * 100
    monthly.index = monthly.index.to_period('M')
    if len(monthly) > 0:
        monthly_df = monthly.to_frame('return')
        monthly_df['year'] = monthly_df.index.year
        monthly_df['month'] = monthly_df.index.month
        pivot = monthly_df.pivot(index='year', columns='month', values='return')
        pivot = pivot.reindex(columns=range(1, 13))
        sns.heatmap(pivot, annot=True, fmt='.1f', cmap='RdYlGn', center=0,
                    cbar_kws={'label': 'Return (%)'}, ax=ax6, linewidths=0.5)
        ax6.set_title('Monthly Returns (%)', fontsize=11)
        ax6.set_xlabel('Month')
        ax6.set_ylabel('Year')

    # 7. Trade P&L Distribution
    ax7 = fig.add_subplot(gs[4, 0])
    if len(trades) > 0:
        pnl = trades["pnl"]
        ax7.hist(pnl[pnl > 0], bins=20, alpha=0.7, color='green', label='Wins', edgecolor='white')
        ax7.hist(pnl[pnl <= 0], bins=20, alpha=0.7, color='red', label='Losses', edgecolor='white')
        ax7.axvline(0, color='black', linestyle='--', alpha=0.5)
        ax7.set_title(f'Trade P&L Distribution | Win Rate: {metrics["win_rate_pct"]:.1f}%', fontsize=11)
        ax7.set_xlabel('P&L ($)')
        ax7.set_ylabel('Frequency')
        ax7.legend()
        ax7.grid(True, alpha=0.3)

    # 8. Cumulative P&L by Signal Type
    ax8 = fig.add_subplot(gs[4, 1])
    if len(trades) > 0 and "signal_type" in trades.columns:
        signal_pnl = trades.groupby("signal_type")["pnl"].sum().sort_values()
        colors = ['green' if x > 0 else 'red' for x in signal_pnl.values]
        signal_pnl.plot(kind='barh', ax=ax8, color=colors, edgecolor='white')
        ax8.set_title('P&L by Signal Type', fontsize=11)
        ax8.set_xlabel('Total P&L ($)')
        ax8.grid(True, alpha=0.3, axis='x')

    # 9. Rolling Sharpe
    ax9 = fig.add_subplot(gs[5, 0])
    window = 252 * 30  # ~1 month of 1-min bars
    rolling_sharpe = returns.rolling(window).apply(
        lambda x: np.sqrt(252 * 390) * x.mean() / x.std() if x.std() > 0 else 0
    )
    ax9.plot(rolling_sharpe.index, rolling_sharpe.values, linewidth=0.8, color='purple')
    ax9.axhline(0, color='black', linestyle='--', alpha=0.5)
    ax9.set_title(f'Rolling Sharpe ({window//390}d window)', fontsize=11)
    ax9.set_ylabel('Sharpe')
    ax9.grid(True, alpha=0.3)

    # 10. Metrics Table
    ax10 = fig.add_subplot(gs[5, 1])
    ax10.axis('off')

    metrics_text = f"""
    Performance Metrics
    {'='*30}
    Total Return:       {metrics['total_return_pct']:>8.2f}%
    Annualized Return:  {metrics['annualized_return_pct']:>8.2f}%
    Sharpe Ratio:       {metrics['sharpe_ratio']:>8.2f}
    Sortino Ratio:      {metrics['sortino_ratio']:>8.2f}
    Calmar Ratio:       {metrics['calmar_ratio']:>8.2f}
    Max Drawdown:       {metrics['max_drawdown_pct']:>8.2f}%
    {'-'*30}
    Total Trades:       {metrics['total_trades']:>8}
    Win Rate:           {metrics['win_rate_pct']:>8.2f}%
    Profit Factor:      {metrics['profit_factor']:>8.2f}
    Expectancy:         {metrics['expectancy']:>8.2f}
    Avg Win:            {metrics['avg_win']:>8.2f}
    Avg Loss:           {metrics['avg_loss']:>8.2f}
    Avg Duration:       {metrics['avg_duration_min']:>8.1f} min
    Final Equity:       ${metrics['final_equity']:>8.2f}
    """

    ax10.text(0.05, 0.95, metrics_text, transform=ax10.transAxes,
              fontsize=10, fontfamily='monospace', verticalalignment='top')

    plt.tight_layout()

    if config.save_path:
        if config.format == "pdf":
            with PdfPages(config.save_path) as pdf:
                pdf.savefig(fig, bbox_inches='tight')
        else:
            fig.savefig(config.save_path, bbox_inches='tight', dpi=300)

    return fig


def create_trade_analysis(
    result: BacktestResult,
    config: Optional[TearsheetConfig] = None
) -> plt.Figure:
    """Create trade analysis charts."""
    if config is None:
        config = TearsheetConfig()

    plt.style.use(config.style)
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    trades = result.trades

    if len(trades) == 0:
        for ax in axes:
            ax.text(0.5, 0.5, 'No trades', ha='center', va='center')
        return fig

    # 1. P&L by hour of day
    ax = axes[0]
    trades["entry_hour"] = trades["entry_time"].dt.hour
    hourly = trades.groupby("entry_hour")["pnl"].mean()
    hourly.plot(kind='bar', ax=ax, color=['green' if x > 0 else 'red' for x in hourly])
    ax.set_title('Avg P&L by Entry Hour')
    ax.set_xlabel('Hour')
    ax.set_ylabel('Avg P&L ($)')
    ax.grid(True, alpha=0.3)

    # 2. P&L by day of week
    ax = axes[1]
    trades["entry_dow"] = trades["entry_time"].dt.dayofweek
    dow_map = {0: 'Mon', 1: 'Tue', 2: 'Wed', 3: 'Thu', 4: 'Fri', 5: 'Sat', 6: 'Sun'}
    dow = trades.groupby("entry_dow")["pnl"].mean()
    dow.index = dow.index.map(dow_map)
    dow.plot(kind='bar', ax=ax, color=['green' if x > 0 else 'red' for x in dow])
    ax.set_title('Avg P&L by Day of Week')
    ax.set_xlabel('Day')
    ax.set_ylabel('Avg P&L ($)')
    ax.grid(True, alpha=0.3)

    # 3. Win rate by signal type
    ax = axes[2]
    if "signal_type" in trades.columns:
        signal_stats = trades.groupby("signal_type").agg(
            win_rate=("pnl", lambda x: (x > 0).mean()),
            count=("pnl", "count")
        ).sort_values("win_rate")
        signal_stats["win_rate"].plot(kind='barh', ax=ax, color='steelblue')
        ax.set_title('Win Rate by Signal Type')
        ax.set_xlabel('Win Rate')
        ax.grid(True, alpha=0.3, axis='x')

    # 4. MAE/MFE
    ax = axes[3]
    # Would need trade-level MAE/MFE data
    ax.text(0.5, 0.5, 'MAE/MFE Analysis\n(requires tick data)', ha='center', va='center')
    ax.set_title('MAE / MFE')

    # 5. Consecutive wins/losses
    ax = axes[4]
    trades["is_win"] = trades["pnl"] > 0
    streaks = trades["is_win"].astype(int).groupby((trades["is_win"] != trades["is_win"].shift()).cumsum()).sum()
    win_streaks = streaks[streaks > 0]
    loss_streaks = -streaks[streaks < 0]
    ax.hist(win_streaks, bins=range(1, win_streaks.max()+2), alpha=0.7, label='Wins', color='green')
    ax.hist(loss_streaks, bins=range(1, loss_streaks.max()+2), alpha=0.7, label='Losses', color='red')
    ax.set_title('Consecutive Wins/Losses')
    ax.set_xlabel('Streak Length')
    ax.set_ylabel('Frequency')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 6. P&L over time
    ax = axes[5]
    trades.set_index("exit_time")["pnl"].cumsum().plot(ax=ax, linewidth=0.8)
    ax.set_title('Cumulative P&L Over Time')
    ax.set_ylabel('Cumulative P&L ($)')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if config.save_path:
        if config.format == "pdf":
            with PdfPages(config.save_path.replace('.pdf', '_trades.pdf')) as pdf:
                pdf.savefig(fig, bbox_inches='tight')
        else:
            fig.savefig(config.save_path.replace('.pdf', '_trades.pdf'), bbox_inches='tight', dpi=300)

    return fig


def generate_report(
    result: BacktestResult,
    output_path: str,
    include_trades: bool = True
):
    """Generate full PDF report."""
    config = TearsheetConfig(save_path=output_path, format="pdf")

    # Main tearsheet
    fig1 = create_tearsheet(result, config)

    # Trade analysis
    if include_trades:
        fig2 = create_trade_analysis(result, config)

    plt.close('all')
    print(f"Report saved to {output_path}")