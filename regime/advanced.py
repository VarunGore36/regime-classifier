"""Advanced regime classifiers using entropy, IV skew, GARCH, shadow gamma."""

from dataclasses import dataclass
from typing: Optional
import pandas as pd
import numpy as np

from .base import RegimeClassifier, RegimeResult
from features import (
    get_entropy_features, classify_entropy_regime,
    calculate_iv_skew, calculate_term_structure, get_skew_signals,
    calculate_garch_volatility, calculate_shadow_gamma, get_garch_signals, get_shadow_gamma_signals,
    calculate_realized_volatility, get_volatility_signals
)
from config import get_config


class EntropyRegimeClassifier(RegimeClassifier):
    """
    Classify regime using multiple entropy measures.

    Low entropy = organized, trending, predictable
    High entropy = disorganized, choppy, random
    """

    def __init__(
        self,
        window: int = 100,
        entropy_type: str = "permutation",  # "shannon", "permutation", "sample", "approximate"
        low_threshold: float = 0.3,
        high_threshold: float = 0.7
    ):
        self.window = window
        self.entropy_type = entropy_type
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold

    def classify(self, data: pd.DataFrame) -> RegimeResult:
        """
        Classify regime using entropy.

        Args:
            data: DataFrame with [timestamp, close] (min 1-min bars)
        """
        returns = np.log(data["close"] / data["close"].shift(1)).dropna()

        entropy_features = get_entropy_features(returns, self.window)

        # Select entropy type
        if self.entropy_type == "shannon":
            entropy = entropy_features.shannon_entropy
        elif self.entropy_type == "permutation":
            entropy = entropy_features.permutation_entropy
        elif self.entropy_type == "sample":
            entropy = entropy_features.sample_entropy
        elif self.entropy_type == "approximate":
            entropy = entropy_features.approximate_entropy
        else:
            entropy = entropy_features.permutation_entropy

        # Normalize to 0-1 over lookback
        lookback = self.window * 5
        entropy_min = entropy.rolling(lookback, min_periods=100).min()
        entropy_max = entropy.rolling(lookback, min_periods=100).max()
        entropy_norm = (entropy - entropy_min) / (entropy_max - entropy_min)

        # Regime
        regime = pd.Series("normal", index=data.index)
        regime[entropy_norm < self.low_threshold] = "low_entropy"
        regime[entropy_norm > self.high_threshold] = "high_entropy"

        # Confidence
        confidence = (entropy_norm - 0.5).abs() * 2  # 0 to 1
        confidence = confidence.clip(0, 1).fillna(0.5)

        components = {
            "shannon_entropy": entropy_features.shannon_entropy,
            "permutation_entropy": entropy_features.permutation_entropy,
            "sample_entropy": entropy_features.sample_entropy,
            "approximate_entropy": entropy_features.approximate_entropy,
            "entropy_normalized": entropy_norm,
            "entropy_regime": entropy_features.entropy_regime
        }

        return RegimeResult(
            regime=regime,
            confidence=confidence,
            components=components
        )


class IVSkewRegimeClassifier(RegimeClassifier):
    """
    Classify regime using IV skew and term structure.

    Steep put skew = fear, hedging demand
    Flat/inverted skew = complacency, call buying
    Term structure slope = forward vol expectations
    """

    def __init__(
        self,
        skew_threshold: float = 0.02,
        kurtosis_threshold: float = 0.01,
        term_threshold: float = 0.0
    ):
        self.skew_threshold = skew_threshold
        self.kurtosis_threshold = kurtosis_threshold
        self.term_threshold = term_threshold

    def classify(self, data: pd.DataFrame) -> RegimeResult:
        """
        Classify regime using IV skew.

        Args:
            data: DataFrame with options chain data in 'options_chain' column
                  and underlying_price series
        """
        # This would need options chain data
        # For now, return neutral if no options data
        if "options_chain" not in data.columns:
            return RegimeResult(
                regime=pd.Series("neutral", index=data.index),
                confidence=pd.Series(0.5, index=data.index),
                components={"error": "No options chain data"}
            )

        # Extract options chain (assuming it's stored as list of DataFrames)
        # This is a simplified version - real implementation would need proper data structure
        pass

        return RegimeResult(
            regime=pd.Series("neutral", index=data.index),
            confidence=pd.Series(0.5, index=data.index),
            components={}
        )


class GARCHRegimeClassifier(RegimeClassifier):
    """
    Classify regime using GARCH conditional volatility.

    GARCH captures volatility clustering and mean reversion.
    """

    def __init__(
        self,
        reestimate_freq: int = 252 * 390,
        high_vol_threshold: float = 0.8,
        low_vol_threshold: float = 0.2
    ):
        self.reestimate_freq = reestimate_freq
        self.high_vol_threshold = high_vol_threshold
        self.low_vol_threshold = low_vol_threshold

    def classify(self, data: pd.DataFrame) -> RegimeResult:
        """
        Classify regime using GARCH volatility.

        Args:
            data: DataFrame with [timestamp, close]
        """
        returns = np.log(data["close"] / data["close"].shift(1)).dropna()

        # Calculate GARCH
        garch_result = calculate_garch_volatility(
            returns,
            reestimate_freq=self.reestimate_freq
        )

        # Align to data index
        vol_regime = garch_result.vol_regime.reindex(data.index, method="ffill")
        conditional_vol = garch_result.conditional_vol.reindex(data.index, method="ffill")
        forecast = garch_result.forecast.reindex(data.index, method="ffill")

        # Confidence based on model fit
        # Higher confidence when vol is clearly in a regime
        vol_percentile = conditional_vol.rolling(252 * 390, min_periods=1000).rank(pct=True)
        confidence = (vol_percentile - 0.5).abs() * 2
        confidence = confidence.clip(0, 1).fillna(0.5)

        components = {
            "conditional_vol": conditional_vol,
            "forecast": forecast,
            "residuals": garch_result.residuals.reindex(data.index, method="ffill"),
            "log_likelihood": garch_result.log_likelihood,
            "params": garch_result.params
        }

        return RegimeResult(
            regime=vol_regime,
            confidence=confidence,
            components=components
        )


class ShadowGammaRegimeClassifier(RegimeClassifier):
    """
    Classify regime using Shadow Gamma (hidden dealer positioning).

    Concentrated gamma = fragile, prone to gamma squeezes
    Dispersed gamma = stable, normal hedging
    """

    def __init__(
        self,
        hhi_threshold: float = 0.3,
        delta_threshold: float = 1e9
    ):
        self.hhi_threshold = hhi_threshold
        self.delta_threshold = delta_threshold

    def classify(self, data: pd.DataFrame) -> RegimeResult:
        """
        Classify regime using Shadow Gamma.

        Args:
            data: DataFrame with options chain and underlying data
        """
        if "options_chain" not in data.columns or "underlying_price" not in data.columns:
            return RegimeResult(
                regime=pd.Series("neutral", index=data.index),
                confidence=pd.Series(0.5, index=data.index),
                components={"error": "Missing options chain or underlying price"}
            )

        # This would need proper options chain data structure
        # Placeholder for now
        return RegimeResult(
            regime=pd.Series("neutral", index=data.index),
            confidence=pd.Series(0.5, index=data.index),
            components={}
        )


class CompositeAdvancedRegimeClassifier(RegimeClassifier):
    """
    Combined advanced regime classifier.

    Combines:
    - Entropy (market organization)
    - GARCH (volatility dynamics)
    - IV Skew (options market sentiment)
    - Shadow Gamma (dealer positioning)
    """

    def __init__(self, weights: Optional[dict] = None):
        default_weights = {
            "entropy": 0.25,
            "garch": 0.25,
            "iv_skew": 0.25,
            "shadow_gamma": 0.25
        }
        self.weights = weights or default_weights

        self.entropy_clf = EntropyRegimeClassifier()
        self.garch_clf = GARCHRegimeClassifier()
        # self.iv_skew_clf = IVSkewRegimeClassifier()
        # self.shadow_gamma_clf = ShadowGammaRegimeClassifier()

    def classify(self, data: pd.DataFrame) -> RegimeResult:
        """Classify using all advanced regime signals."""
        # Entropy
        entropy_result = self.entropy_clf.classify(data)

        # GARCH
        garch_result = self.garch_clf.classify(data)

        # Combine regimes
        entropy_score = self._map_entropy_regime(entropy_result.regime)
        garch_score = self._map_garch_regime(garch_result.regime)

        # Weighted composite
        composite_score = (
            entropy_score * self.weights["entropy"] +
            garch_score * self.weights["garch"]
        )

        # Primary regime
        regime = pd.Series("neutral", index=data.index)
        regime[composite_score > 0.2] = "favorable"
        regime[composite_score < -0.2] = "unfavorable"

        # Confidence
        confidence = (
            entropy_result.confidence * self.weights["entropy"] +
            garch_result.confidence * self.weights["garch"]
        )

        components = {
            "entropy": entropy_result.components,
            "garch": garch_result.components,
            "composite_score": composite_score
        }

        return RegimeResult(
            regime=regime,
            confidence=confidence,
            components=components
        )

    def _map_entropy_regime(self, regime: pd.Series) -> pd.Series:
        mapping = {
            "low_entropy": 0.5,    # Organized, trending
            "high_entropy": -0.5,  # Disorganized, choppy
            "normal": 0.0
        }
        return regime.map(mapping).fillna(0.0)

    def _map_garch_regime(self, regime: pd.Series) -> pd.Series:
        mapping = {
            "low_vol": 0.3,        # Stable
            "high_vol": -0.3,      # Unstable
            "normal": 0.0
        }
        return regime.map(mapping).fillna(0.0)


def get_advanced_regime_summary(result: RegimeResult, timestamp: pd.Timestamp) -> dict:
    """Get summary of advanced regime at timestamp."""
    idx = result.regime.index.searchsorted(timestamp) - 1
    if idx < 0:
        return {}

    return {
        "regime": result.regime.iloc[idx],
        "confidence": result.confidence.iloc[idx],
        "composite_score": result.components.get("composite_score", pd.Series()).iloc[idx] if "composite_score" in result.components else None,
        "entropy_regime": result.components.get("entropy", {}).get("entropy_regime", pd.Series()).iloc[idx] if "entropy" in result.components else None,
        "garch_regime": result.components.get("garch", {}).get("vol_regime", pd.Series()).iloc[idx] if "garch" in result.components else None,
    }