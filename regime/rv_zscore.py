"""RV Z-Score based regime classifier."""

from dataclasses import dataclass
from typing: Optional
import pandas as pd
import numpy as np

from .base import RegimeClassifier, RegimeResult
from features import calculate_realized_volatility, get_volatility_signals
from config import get_config


class RVZScoreRegimeClassifier(RegimeClassifier):
    """
    Classify regime based on Realized Volatility Z-Score.

    Regimes:
    - low_vol: Z < 0 (vol below historical average)
    - normal_vol: 0 <= Z < 1 (normal vol)
    - elevated_vol: 1 <= Z < 2 (elevated vol)
    - very_elevated_vol: Z >= 2 (very elevated vol)
    """

    def __init__(
        self,
        window_minutes: Optional[int] = None,
        zscore_windows: Optional[list] = None,
        elevated_threshold: Optional[float] = None,
        very_elevated_threshold: Optional[float] = None,
        min_confidence: float = 0.6
    ):
        config = get_config()
        self.window_minutes = window_minutes or config.realized_vol.window_minutes
        self.zscore_windows = zscore_windows or config.realized_vol.zscore_windows
        self.elevated_threshold = elevated_threshold or config.realized_vol.elevated_threshold
        self.very_elevated_threshold = very_elevated_threshold or config.realized_vol.very_elevated_threshold
        self.min_confidence = min_confidence

    def classify(self, data: pd.DataFrame) -> RegimeResult:
        """
        Classify regime based on RV Z-score.

        Args:
            data: DataFrame with [timestamp, close] (min 1-min bars)
        """
        rv_result = calculate_realized_volatility(
            data,
            window_minutes=self.window_minutes,
            zscore_windows=self.zscore_windows,
            elevated_threshold=self.elevated_threshold,
            very_elevated_threshold=self.very_elevated_threshold
        )

        signals = get_volatility_signals(rv_result, self.elevated_threshold)

        # Confidence based on how far Z-score is from boundaries
        z = rv_result.rv_zscore
        confidence = pd.Series(0.5, index=data.index)

        # High confidence when clearly in a regime
        confidence[z < -0.5] = 0.8  # Clearly low vol
        confidence[(z >= -0.5) & (z < 0.5)] = 0.6  # Near normal
        confidence[(z >= 0.5) & (z < 1.5)] = 0.7  # Clearly elevated
        confidence[z >= 1.5] = 0.9  # Very elevated

        # Low confidence in transition zones
        confidence[(z >= -0.5) & (z < 0.5)] = 0.5

        components = {
            "rv": rv_result.rv,
            "rv_zscore": rv_result.rv_zscore,
            "vol_acceleration": rv_result.volatility_acceleration,
            "vol_regime": rv_result.regime,
            "signals": signals
        }

        return RegimeResult(
            regime=rv_result.regime,
            confidence=confidence,
            components=components
        )


class GEXRegimeClassifier(RegimeClassifier):
    """
    Classify regime based on GEX (Gamma Exposure) sign.

    Regimes:
    - positive_gex: Dealers long gamma (mean reversion)
    - negative_gex: Dealers short gamma (trend following)
    - neutral_gex: Near gamma flip
    """

    def __init__(
        self,
        gex_positive_threshold: float = 0,
        gex_negative_threshold: float = 0,
        flip_buffer: float = 0.1
    ):
        self.gex_positive_threshold = gex_positive_threshold
        self.gex_negative_threshold = gex_negative_threshold
        self.flip_buffer = flip_buffer

    def classify(self, data: pd.DataFrame) -> RegimeResult:
        """
        Classify regime based on GEX.

        Args:
            data: DataFrame with [timestamp, gex_net, gamma_flip] columns
        """
        if "gex_net" not in data.columns:
            raise ValueError("Data must contain 'gex_net' column")

        gex = data["gex_net"]
        gamma_flip = data.get("gamma_flip", pd.Series(index=data.index, dtype=float))

        # Distance from gamma flip
        if "underlying_price" in data.columns:
            dist_from_flip = (data["underlying_price"] - gamma_flip).abs()
        else:
            dist_from_flip = pd.Series(1.0, index=data.index)

        # Regime classification
        regime = pd.Series("neutral_gex", index=data.index)
        regime[gex > self.gex_positive_threshold] = "positive_gex"
        regime[gex < self.gex_negative_threshold] = "negative_gex"

        # Near flip = neutral
        near_flip = dist_from_flip < (dist_from_flip.rolling(20).mean() * self.flip_buffer)
        regime[near_flip] = "neutral_gex"

        # Confidence based on GEX magnitude
        confidence = (gex.abs() / gex.abs().rolling(100).quantile(0.9)).clip(0, 1)
        confidence = confidence.fillna(0.5)

        components = {
            "gex_net": gex,
            "gamma_flip": gamma_flip,
            "dist_from_flip": dist_from_flip
        }

        return RegimeResult(
            regime=regime,
            confidence=confidence,
            components=components
        )


class DEXRegimeClassifier(RegimeClassifier):
    """
    Classify regime based on DEX (Delta Exposure).

    Regimes:
    - long_dex: Dealers net long delta
    - short_dex: Dealers net short delta
    - neutral_dex: Near zero
    """

    def __init__(
        self,
        dex_positive_threshold: float = 0,
        dex_negative_threshold: float = 0
    ):
        self.dex_positive_threshold = dex_positive_threshold
        self.dex_negative_threshold = dex_negative_threshold

    def classify(self, data: pd.DataFrame) -> RegimeResult:
        """
        Classify regime based on DEX.

        Args:
            data: DataFrame with [timestamp, dex_net] column
        """
        if "dex_net" not in data.columns:
            raise ValueError("Data must contain 'dex_net' column")

        dex = data["dex_net"]

        regime = pd.Series("neutral_dex", index=data.index)
        regime[dex > self.dex_positive_threshold] = "long_dex"
        regime[dex < self.dex_negative_threshold] = "short_dex"

        confidence = (dex.abs() / dex.abs().rolling(100).quantile(0.9)).clip(0, 1)
        confidence = confidence.fillna(0.5)

        components = {"dex_net": dex}

        return RegimeResult(
            regime=regime,
            confidence=confidence,
            components=components
        )


class EntropyRegimeClassifier(RegimeClassifier):
    """
    Classify regime based on market entropy (placeholder).

    This is a placeholder for the entropy model mentioned in the paper.
    """

    def __init__(self, window: int = 100):
        self.window = window

    def calculate_entropy(self, returns: pd.Series) -> pd.Series:
        """Calculate Shannon entropy of return distribution."""
        # Discretize returns into bins
        bins = 10
        entropy = pd.Series(index=returns.index, dtype=float)

        for i in range(self.window, len(returns)):
            window_returns = returns.iloc[i-self.window:i]
            hist, _ = np.histogram(window_returns.dropna(), bins=bins, density=True)
            hist = hist[hist > 0]
            entropy.iloc[i] = -np.sum(hist * np.log(hist))

        return entropy

    def classify(self, data: pd.DataFrame) -> RegimeResult:
        """
        Classify regime based on entropy.

        Args:
            data: DataFrame with [timestamp, close]
        """
        returns = np.log(data["close"] / data["close"].shift(1))
        entropy = self.calculate_entropy(returns)

        # Normalize entropy
        entropy_norm = (entropy - entropy.rolling(252).mean()) / entropy.rolling(252).std()

        # High entropy = disorganized/uncertain
        # Low entropy = organized/trending
        regime = pd.Series("normal", index=data.index)
        regime[entropy_norm > 1] = "high_entropy"
        regime[entropy_norm < -1] = "low_entropy"

        confidence = entropy_norm.abs().clip(0, 1).fillna(0.5)

        components = {
            "entropy": entropy,
            "entropy_normalized": entropy_norm
        }

        return RegimeResult(
            regime=regime,
            confidence=confidence,
            components=components
        )