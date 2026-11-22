"""Combined signal generator."""

from dataclasses import dataclass
from typing: Optional, List
import pandas as pd
import numpy as np

from config import get_config
from features import (
    calculate_volume_profile, calculate_vwap, get_profile_levels,
    get_entropy_features, calculate_iv_skew, calculate_term_structure,
    get_skew_signals, calculate_garch_volatility, get_garch_signals,
    calculate_shadow_gamma, get_shadow_gamma_signals
)
from regime import CombinedRegimeClassifier, CompositeAdvancedRegimeClassifier
from .entries import POCReversionSignal, VWAPBandSignal, BreakFadeSignal, EntrySignal
from .exits import StopLossExit, TakeProfitExit, TrailingStopExit, TimeExit, RegimeChangeExit, combine_exits, ExitSignal
from .advanced import (
    EntropySignalGenerator, IVSkewSignalGenerator, GARCHSignalGenerator, ShadowGammaSignalGenerator
)


@dataclass
class TradingSignal:
    """Complete trading signal with entry and exit."""
    entry: EntrySignal
    exit: ExitSignal
    position: pd.Series           # 1=long, -1=short, 0=flat
    equity_curve: pd.Series       # Cumulative P&L
    regime_result: any            # Regime classification result
    advanced_signals: dict        # Advanced signal results


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

        # Advanced signals
        self.entropy_signal = EntropySignalGenerator(config.get("entropy_signal"))
        self.iv_skew_signal = IVSkewSignalGenerator(config.get("iv_skew_signal"))
        self.garch_signal = GARCHSignalGenerator(config.get("garch_signal"))
        self.shadow_gamma_signal = ShadowGammaSignalGenerator(config.get("shadow_gamma_signal"))

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
            exit_hour=getattr(cfg.backtest, "session_close_hour", 15),
            exit_minute=getattr(cfg.backtest, "session_close_minute", 55)
        )

        # Regime classifiers
        self.regime_classifier = CombinedRegimeClassifier()
        self.advanced_regime_classifier = CompositeAdvancedRegimeClassifier()

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

        # 2. Advanced features
        returns = np.log(nq_bars["close"] / nq_bars["close"].shift(1)).dropna()
        entropy_features = get_entropy_features(returns)

        # 3. Prepare regime data
        regime_data = self._prepare_regime_data(nq_bars, options_chain)

        # 4. Classify regimes
        regime_result = self.regime_classifier.classify(regime_data)
        advanced_regime_result = self.advanced_regime_classifier.classify(regime_data)

        # 5. Generate entry signals (basic)
        poc_entry = self.poc_signal.generate(nq_bars, vol_profile, regime_result)
        vwap_entry = self.vwap_signal.generate(nq_bars, vwap_result, regime_result)
        bf_entry = self.break_fade_signal.generate(nq_bars, footprint, profile_levels, regime_result)

        # 6. Generate advanced signals
        entropy_entry = self.entropy_signal.generate(nq_bars, entropy_features, advanced_regime_result)

        iv_skew_entry = IVSkewSignal(
            long_entry=pd.Series(False, index=nq_bars.index),
            short_entry=pd.Series(False, index=nq_bars.index),
            long_price=pd.Series(np.nan, index=nq_bars.index),
            short_price=pd.Series(np.nan, index=nq_bars.index),
            stop_loss=pd.Series(np.nan, index=nq_bars.index),
            take_profit=pd.Series(np.nan, index=nq_bars.index),
            signal_type=pd.Series("none", index=nq_bars.index),
            metadata={}
        )
        garch_entry = GARCHSignal(
            long_entry=pd.Series(False, index=nq_bars.index),
            short_entry=pd.Series(False, index=nq_bars.index),
            long_price=pd.Series(np.nan, index=nq_bars.index),
            short_price=pd.Series(np.nan, index=nq_bars.index),
            stop_loss=pd.Series(np.nan, index=nq_bars.index),
            take_profit=pd.Series(np.nan, index=nq_bars.index),
            signal_type=pd.Series("none", index=nq_bars.index),
            metadata={}
        )
        shadow_gamma_entry = ShadowGammaSignal(
            long_entry=pd.Series(False, index=nq_bars.index),
            short_entry=pd.Series(False, index=nq_bars.index),
            long_price=pd.Series(np.nan, index=nq_bars.index),
            short_price=pd.Series(np.nan, index=nq_bars.index),
            stop_loss=pd.Series(np.nan, index=nq_bars.index),
            take_profit=pd.Series(np.nan, index=nq_bars.index),
            signal_type=pd.Series("none", index=nq_bars.index),
            metadata={}
        )

        # Try to generate advanced signals if options data available
        if len(options_chain) > 0:
            try:
                # IV Skew
                skew_result = calculate_iv_skew(
                    options_chain, nq_bars.set_index("timestamp")["close"]
                )
                term_result = calculate_term_structure(options_chain)
                skew_signals = get_skew_signals(skew_result, term_result)
                iv_skew_entry = self.iv_skew_signal.generate(nq_bars, skew_signals, advanced_regime_result)

                # GARCH
                garch_result = calculate_garch_volatility(returns)
                garch_signals = get_garch_signals(garch_result, returns)
                garch_entry = self.garch_signal.generate(nq_bars, garch_signals, advanced_regime_result)

                # Shadow Gamma
                shadow_result = calculate_shadow_gamma(options_chain, nq_bars.set_index("timestamp")["close"])
                shadow_signals = get_shadow_gamma_signals(shadow_result)
                shadow_gamma_entry = self.shadow_gamma_signal.generate(nq_bars, shadow_signals, advanced_regime_result)
            except Exception as e:
                print(f"Advanced signals error: {e}")

        # 7. Combine all entries (priority order)
        all_entries = [
            poc_entry,      # Highest priority
            vwap_entry,
            bf_entry,
            entropy_entry,
            iv_skew_entry,
            garch_entry,
            shadow_gamma_entry
        ]
        combined_entry = self._combine_entries(all_entries)

        # 8. Generate positions from entries
        position = self._entries_to_position(nq_bars, combined_entry)

        # 9. Generate exit signals
        entry_price = combined_entry.long_price.where(
            combined_entry.long_entry,
            combined_entry.short_price.where(combined_entry.short_entry)
        )
        exits = []
        exits.append(self.stop_loss.generate(nq_bars, entry_price, position))
        exits.append(self.take_profit.generate(nq_bars, entry_price, position))
        exits.append(self.trailing_stop.generate(nq_bars, entry_price, position))
        exits.append(self.time_exit.generate(nq_bars, position))
        exits.append(RegimeChangeExit(self.regime_classifier).generate(nq_bars, position, regime_result))

        combined_exit = combine_exits(*exits)

        # 10. Apply exits to position
        final_position = self._apply_exits(position, combined_exit)

        # 11. Calculate equity curve
        equity = self._calculate_equity(nq_bars, final_position, combined_entry, combined_exit)

        # 12. Advanced signals dict
        advanced_signals = {
            "entropy": entropy_entry,
            "iv_skew": iv_skew_entry,
            "garch": garch_entry,
            "shadow_gamma": shadow_gamma_entry
        }

        return TradingSignal(
            entry=combined_entry,
            exit=combined_exit,
            position=final_position,
            equity_curve=equity,
            regime_result=regime_result,
            advanced_signals=advanced_signals
        )

    def _prepare_regime_data(self, nq_bars: pd.DataFrame, options_chain: pd.DataFrame) -> pd.DataFrame:
        """Prepare data for regime classification."""
        df = nq_bars[["timestamp", "close"]].copy()

        # Add GEX/DEX from options chain
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

        # Add IV/RV placeholder
        df["iv"] = np.nan
        df["rv"] = np.nan

        # Add options_chain for advanced classifiers
        df["options_chain"] = [options_chain] * len(df) if len(options_chain) > 0 else None

        return df

    def _combine_entries(self, entries: List[EntrySignal]) -> EntrySignal:
        """Combine multiple entry signals with priority."""
        if not entries:
            return entries[0] if entries else None

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
                if entry.long_entry.iloc[i]:
                    position.iloc[i] = 1
                elif entry.short_entry.iloc[i]:
                    position.iloc[i] = -1
                else:
                    position.iloc[i] = 0
            else:
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

            if prev_position == 0 and curr_position != 0:
                in_position = True
                position_side = curr_position
                if curr_position == 1:
                    entry_price = bars["close"].iloc[i] + slippage
                else:
                    entry_price = bars["close"].iloc[i] - slippage
                prev_equity -= commission

            elif prev_position != 0 and curr_position == 0:
                in_position = False
                if position_side == 1:
                    exit_price = bars["close"].iloc[i] - slippage
                    trade_pnl = (exit_price - entry_price) * contract_multiplier
                else:
                    exit_price = bars["close"].iloc[i] + slippage
                    trade_pnl = (entry_price - exit_price) * contract_multiplier
                trade_pnl -= commission
                prev_equity += trade_pnl
                position_side = 0

            if in_position:
                if position_side == 1:
                    unrealized = (bars["close"].iloc[i] - entry_price) * contract_multiplier
                else:
                    unrealized = (entry_price - bars["close"].iloc[i]) * contract_multiplier
                equity.iloc[i] = prev_equity + unrealized
            else:
                equity.iloc[i] = prev_equity

        return equity