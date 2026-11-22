"""Signals package."""

from .entries import POCReversionSignal, VWAPBandSignal, BreakFadeSignal, EntrySignal
from .exits import StopLossExit, TakeProfitExit, TrailingStopExit, TimeExit, RegimeChangeExit, ExitSignal, combine_exits
from .generator import SignalGenerator, TradingSignal
from .advanced import (
    EntropySignalGenerator, IVSkewSignalGenerator, GARCHSignalGenerator, ShadowGammaSignalGenerator,
    EntropySignal, IVSkewSignal, GARCHSignal, ShadowGammaSignal
)

__all__ = [
    # Entries
    "POCReversionSignal", "VWAPBandSignal", "BreakFadeSignal", "EntrySignal",
    # Exits
    "StopLossExit", "TakeProfitExit", "TrailingStopExit", "TimeExit", "RegimeChangeExit", "ExitSignal", "combine_exits",
    # Generator
    "SignalGenerator", "TradingSignal",
    # Advanced
    "EntropySignalGenerator", "IVSkewSignalGenerator", "GARCHSignalGenerator", "ShadowGammaSignalGenerator",
    "EntropySignal", "IVSkewSignal", "GARCHSignal", "ShadowGammaSignal",
]