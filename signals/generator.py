"""Combined signal generator."""

from dataclasses import dataclass
from typing: Optional, List
import pandas as pd
import numpy as np

from config import get_config
from features import calculate_volume_profile, calculate_vwap, get_profile_levels
from regime import CombinedRegimeClassifier
from .entries import POCReversionSignal, VWAPBandSignal, BreakFadeSignal, EntrySignal
from .exits import StopLossExit, TakeProfitExit, TrailingStopExit, TimeExit, RegimeChangeExit, combine_exits, ExitSignal


@dataclass
class TradingSignal:
    """Complete trading signal with entry and exit."""
    entry: EntrySignal
    exit: ExitSignal
    position: pd.Series           # 1=long, -1=short, 0=flat
    equity_curve: pd.Series       # Cumulative P&L


class SignalGenerator:
    """
    Main signal generator combining all entry/exit logic.
    """

    def __init__(self, config: Optional[dict] = None):
        self.config = config or {}
        cfg = get_config()

        # Entry signals
        self.poc_signal = POCReversionSignal(config.get("poc_reversion"))
        self.vwap_signal = VWAPBandSignal(config.get("vwap_bands"))
        self.break_fade_signal = BreakFadeSignal(config.get("break_fade"))

        # Exit signals
        self.stop_loss = StopLossExit(
            stop_ticks=config.get("stop_ticks", 20),
            tick_size=0.25
        )
        self.take_profit = TakeProfitExit(
            target_ticks=config.get("target_ticks", 40),
            tick_size=0.25
        )
        self.trailing_stop = TrailingStopExit(
            trail_ticks=config.get("trail_ticks", 10),
            tick_size=0.25
        )
        self.time_exit = TimeExit(
            exit_hour=cfg.backtest.session_close_hour if hasattr(cfg.backtest, "session_close_hour") else 15,
            exit_minute=cfg.backtest.session_close_minute if hasattr(cfg.backtest, "session_close_minute") else 55
        )

        # Regime classifier
        self.regime_classifier = CombinedRegimeClassifier()

    def generate(
        self,
        nq_bars: pd.DataFrame,
        qqq_bars: pd.DataFrame,
        options_chain: pd.DataFrame,
        footprint: pd.DataFrame
    ) -> TradingSignal:
        """
        Generate complete trading signals.

        Args:
            nq_bars: NQ OHLCV bars
            qqq_bars: QQQ OHLCV bars
            options_chain: Options chain data
            footprint: Footprint/order flow data

        Returns:
            TradingSignal with entries, exits, positions, equity
        """
        # 1. Calculate features
        vol_profile = calculate_volume_profile(nq_bars)
        vwap_result = calculate_vwap(nq_bars)
        profile_levels = get_profile_levels(vol_profile)

        # 2. Prepare regime data
        regime_data = self._prepare_regime_data(nq_bars, options_chain)

        # 3. Classify regime
        regime_result = self.regime_classifier.classify(regime_data)

        # 4. Generate entry signals
        poc_entry = self.poc_signal.generate(nq_bars, vol_profile, regime_result)
        vwap_entry = self.vwap_signal.generate(nq_bars, vwap_result, regime_result)
        bf_entry = self.break_fade_signal.generate(nq_bars, footprint, profile_levels, regime_result)

        # 5. Combine entries (priority: POC > VWAP > Break/Fade)
        combined_entry = self._combine_entries([poc_entry, vwap_entry, bf_entry])

        # 6. Generate positions from entries
        position = self._entries_to_position(nq_bars, combined_entry)

        # 7. Generate exit signals
        exits = []
        exits.append(self.stop_loss.generate(nq_bars, combined_entry.long_price.where(combined_entry.long_entry, combined_entry.short_price.where(combined_entry.short_entry)), position))
        exits.append(self.take_profit.generate(nq_bars, combined_entry.long_price.where(combined_entry.long_entry, combined_entry.short_price.where(combined_entry.short_entry)), position))
        exits.append(self.trailing_stop.generate(nq_bars, combined_entry.long_price.where(combined_entry.long_entry, combined_entry.short_price.where(combined_entry.short_entry)), position))
        exits.append(self.time_exit.generate(nq_bars, position))
        exits.append(RegimeChangeExit(self.regime_classifier).generate(nq_bars, position, regime_result))

        combined_exit = combine_exits(*exits)

        # 8. Apply exits to position
        final_position = self._apply_exits(position, combined_exit)

        # 9. Calculate equity curve
        equity = self._calculate_equity(nq_bars, final_position, combined_entry, combined_exit)

        return TradingSignal(
            entry=combined_entry,
            exit=combined_exit,
            position=final_position,
            equity_curve=equity
        )

    def _prepare_regime_data(self, nq_bars: pd.DataFrame, options_chain: pd.DataFrame) -> pd.DataFrame:
        """Prepare data for regime classification."""
        df = nq_bars[["timestamp", "close"]].copy()

        # Add GEX/DEX from options chain (simplified - would need proper aggregation)
        if len(options_chain) > 0:
            # Get GEX/DEX per timestamp from options
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

        # Add IV/RV if available
        df["iv"] = np.nan  # Would come from IV features
        df["rv"] = np.nan  # Would come from RV features

        return df

    def _combine_entries(self, entries: List[EntrySignal]) -> EntrySignal:
        """Combine multiple entry signals with priority."""
        if not entries:
            return entries[0] if entries else None

        # Use first entry as base
        base = entries[0]
        n = len(base.long_entry)

        combined_long = base.long_entry.copy()
        combined_short = base.short_entry.copy()
        combined_long_price = base.long_price.copy()
        combined_short_price = base.short_price.copy()
        combined_stop = base.stop_loss.copy()
        combined_target = base.take_profit.copy()
        combined_type = base.signal_type.copy()

        for entry in entries[1:]:
            # Only add signals where no signal exists yet
            long_mask = entry.long_entry & ~combined_long
            short_mask = entry.short_entry & ~combined_short

            combined_long[long_mask] = True
            combined_short[short_mask] = True
            combined_long_price[long_mask] = entry.long_price[long_mask]
            combined_short_price[short_mask] = entry.short_price[short_mask]
            combined_stop[long_mask] = entry.stop_loss[long_mask]
            combined_stop[short_mask] = entry.stop_loss[short_mask]
            combined_target[long_mask] = entry.take_profit[long_mask]
            combined_target[short_mask] = entry.take_profit[short_mask]
            combined_type[long_mask] = entry.signal_type[long_mask]
            combined_type[short_mask] = entry.signal_type[short_mask]

        return EntrySignal(
            long_entry=combined_long,
            short_entry=combined_short,
            long_price=combined_long_price,
            short_price=combined_short_price,
            stop_loss=combined_stop,
            take_profit=combined_target,
            signal_type=combined_type,
            metadata={"combined": True}
        )

    def _entries_to_position(self, bars: pd.DataFrame, entry: EntrySignal) -> pd.Series:
        """Convert entry signals to position series."""
        position = pd.Series(0, index=bars.index, dtype=int)

        for i in range(len(bars)):
            if i == 0:
                prev_pos = 0
            else:
                prev_pos = position.iloc[i-1]

            if prev_pos == 0:
                # Flat - check for new entry
                if entry.long_entry.iloc[i]:
                    position.iloc[i] = 1
                elif entry.short_entry.iloc[i]:
                    position.iloc[i] = -1
                else:
                    position.iloc[i] = 0
            else:
                # In position - hold until exit (handled separately)
                position.iloc[i] = prev_pos

        return position

    def _apply_exits(self, position: pd.Series, exit: ExitSignal) -> pd.Series:
        """Apply exit signals to position."""
        final_pos = position.copy()

        for i in range(len(position)):
            if i == 0:
                continue

            if final_pos.iloc[i-1] == 1 and exit.exit_long.iloc[i]:
                final_pos.iloc[i] = 0
            elif final_pos.iloc[i-1] == -1 and exit.exit_short.iloc[i]:
                final_pos.iloc[i] = 0
            else:
                final_pos.iloc[i] = final_pos.iloc[i-1]

        return final_pos

    def _calculate_equity(
        self,
        bars: pd.DataFrame,
        position: pd.Series,
        entry: EntrySignal,
        exit: ExitSignal
    ) -> pd.Series:
        """Calculate equity curve from positions."""
        cfg = get_config()
        contract_multiplier = cfg.backtest.contract_multiplier
        commission = cfg.backtest.commission_per_contract
        slippage = cfg.backtest.slippage_ticks * 0.25

        equity = pd.Series(0.0, index=bars.index)
        pnl = pd.Series(0.0, index=bars.index)

        in_position = False
        entry_price = 0
        position_side = 0

        for i in range(len(bars)):
            if i == 0:
                equity.iloc[i] = cfg.backtest.initial_capital
                continue

            prev_equity = equity.iloc[i-1]
            curr_position = position.iloc[i]
            prev_position = position.iloc[i-1]

            # Entry
            if prev_position == 0 and curr_position != 0:
                in_position = True
                position_side = curr_position
                if curr_position == 1:
                    entry_price = bars["close"].iloc[i] + slippage
                else:
                    entry_price = bars["close"].iloc[i] - slippage

                # Commission
                trade_cost = commission
                prev_equity -= trade_cost

            # Exit
            elif prev_position != 0 and curr_position == 0:
                in_position = False
                if position_side == 1:
                    exit_price = bars["close"].iloc[i] - slippage
                    trade_pnl = (exit_price - entry_price) * contract_multiplier
                else:
                    exit_price = bars["close"].iloc[i] + slippage
                    trade_pnl = (entry_price - exit_price) * contract_multiplier

                # Commission
                trade_cost = commission
                trade_pnl -= trade_cost

                prev_equity += trade_pnl
                pnl.iloc[i] = trade_pnl
                position_side = 0

            # Mark-to-market (unrealized P&L)
            if in_position:
                if position_side == 1:
                    unrealized = (bars["close"].iloc[i] - entry_price) * contract_multiplier
                else:
                    unrealized = (entry_price - bars["close"].iloc[i]) * contract_multiplier
                equity.iloc[i] = prev_equity + unrealized
            else:
                equity.iloc[i] = prev_equity

        return equity