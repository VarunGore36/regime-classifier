"""Combined regime classifier using multiple signals."""

from dataclasses import dataclass
from typing: Optional, Dict
import pandas as pd
import numpy as np

from .base import RegimeClassifier, RegimeResult, CompositeRegimeClassifier
from .rv_zscore import RVZScoreRegimeClassifier, GEXRegimeClassifier, DEXRegimeClassifier, EntropyRegimeClassifier
from config import get_config


@dataclass
class CombinedRegimeResult:
    """Result from combined regime classification."""
    primary_regime: pd.Series           # Primary regime label
    regime_score: pd.Series             # Composite score (-1 to 1)
    confidence: pd.Series               # Overall confidence
    sub_regimes: Dict[str, pd.Series]   # Individual classifier regimes
    bias: pd.Series                     # Directional bias: bullish/bearish/neutral
    components: Dict                    # All component data


class CombinedRegimeClassifier(RegimeClassifier):
    """
    Combined regime classifier using multiple signals:
    - RV Z-score (volatility environment)
    - GEX sign (dealer gamma positioning)
    - DEX sign (dealer delta positioning)
    - Entropy (market organization)
    - IV/RV ratio (vol risk premium)
    """

    def __init__(self, config: Optional[dict] = None):
        self.config = config or {}
        cfg = get_config()

        # Initialize sub-classifiers
        self.rv_classifier = RVZScoreRegimeClassifier(
            window_minutes=cfg.realized_vol.window_minutes,
            zscore_windows=cfg.realized_vol.zscore_windows,
            elevated_threshold=cfg.realized_vol.elevated_threshold,
            very_elevated_threshold=cfg.realized_vol.very_elevated_threshold,
            min_confidence=cfg.regime.min_confidence
        )
        self.gex_classifier = GEXRegimeClassifier(
            gex_positive_threshold=cfg.greeks.gex_positive_threshold,
            gex_negative_threshold=cfg.greeks.gex_negative_threshold
        )
        self.dex_classifier = DEXRegimeClassifier(
            dex_positive_threshold=cfg.greeks.dex_positive_threshold,
            dex_negative_threshold=cfg.greeks.dex_negative_threshold
        )
        self.entropy_classifier = EntropyRegimeClassifier()

        # Weights for composite
        self.weights = {
            "rv": 0.30,
            "gex": 0.25,
            "dex": 0.20,
            "entropy": 0.15,
            "iv_rv": 0.10
        }

    def classify(self, data: pd.DataFrame) -> CombinedRegimeResult:
        """
        Classify combined regime.

        Args:
            data: DataFrame with columns:
                - timestamp, close (required)
                - gex_net, gamma_flip, underlying_price (for GEX)
                - dex_net (for DEX)
                - iv, rv (for IV/RV ratio)
        """
        # Get sub-regime classifications
        rv_result = self.rv_classifier.classify(data)
        gex_result = self.gex_classifier.classify(data)
        dex_result = self.dex_classifier.classify(data)
        entropy_result = self.entropy_classifier.classify(data)

        # IV/RV regime
        iv_rv_regime = pd.Series("neutral", index=data.index)
        iv_rv_confidence = pd.Series(0.5, index=data.index)
        if "iv" in data.columns and "rv" in data.columns:
            iv_rv = data["iv"] / data["rv"]
            iv_rv_regime[iv_rv > 1.2] = "iv_expensive"
            iv_rv_regime[iv_rv < 0.8] = "iv_cheap"
            iv_rv_confidence = (iv_rv - 1).abs().clip(0, 1).fillna(0.5)

        # Combine regimes into composite score
        # Map regimes to numeric scores
        rv_score = self._map_rv_regime(rv_result.regime)
        gex_score = self._map_gex_regime(gex_result.regime)
        dex_score = self._map_dex_regime(dex_result.regime)
        entropy_score = self._map_entropy_regime(entropy_result.regime)
        iv_rv_score = self._map_iv_rv_regime(iv_rv_regime)

        # Weighted composite score
        composite_score = (
            rv_score * self.weights["rv"] +
            gex_score * self.weights["gex"] +
            dex_score * self.weights["dex"] +
            entropy_score * self.weights["entropy"] +
            iv_rv_score * self.weights["iv_rv"]
        )

        # Primary regime from composite score
        primary_regime = pd.Series("neutral", index=data.index)
        primary_regime[composite_score > 0.3] = "bullish_environment"
        primary_regime[composite_score < -0.3] = "bearish_environment"
        primary_regime[(composite_score >= -0.3) & (composite_score <= 0.3)] = "neutral_environment"

        # High vol regime overlay
        primary_regime[rv_result.regime == "very_elevated_vol"] = "high_vol_" + primary_regime[rv_result.regime == "very_elevated_vol"]
        primary_regime[rv_result.regime == "elevated_vol"] = "elevated_vol_" + primary_regime[rv_result.regime == "elevated_vol"]
        primary_regime[rv_result.regime == "low_vol"] = "low_vol_" + primary_regime[rv_result.regime == "low_vol"]

        # Directional bias
        bias = pd.Series("neutral", index=data.index)
        bias[composite_score > 0.1] = "bullish"
        bias[composite_score < -0.1] = "bearish"

        # Overall confidence (weighted average)
        confidence = (
            rv_result.confidence * self.weights["rv"] +
            gex_result.confidence * self.weights["gex"] +
            dex_result.confidence * self.weights["dex"] +
            entropy_result.confidence * self.weights["entropy"] +
            iv_rv_confidence * self.weights["iv_rv"]
        )

        # Sub-regimes dict
        sub_regimes = {
            "rv": rv_result.regime,
            "gex": gex_result.regime,
            "dex": dex_result.regime,
            "entropy": entropy_result.regime,
            "iv_rv": iv_rv_regime
        }

        # All components
        components = {
            "rv": rv_result.components,
            "gex": gex_result.components,
            "dex": dex_result.components,
            "entropy": entropy_result.components,
            "iv_rv": {"ratio": data.get("iv", pd.Series()) / data.get("rv", pd.Series())},
            "composite_score": composite_score
        }

        return CombinedRegimeResult(
            primary_regime=primary_regime,
            regime_score=composite_score,
            confidence=confidence,
            sub_regimes=sub_regimes,
            bias=bias,
            components=components
        )

    def _map_rv_regime(self, regime: pd.Series) -> pd.Series:
        """Map RV regime to numeric score."""
        mapping = {
            "very_elevated_vol": -0.5,  # High vol = risky
            "elevated_vol": -0.2,
            "normal_vol": 0.1,
            "low_vol": 0.3,  # Low vol = stable
            "unknown": 0.0
        }
        return regime.map(mapping).fillna(0.0)

    def _map_gex_regime(self, regime: pd.Series) -> pd.Series:
        """Map GEX regime to numeric score."""
        mapping = {
            "positive_gex": 0.5,   # Mean reversion friendly
            "negative_gex": -0.5,  # Trend following, risky
            "neutral_gex": 0.0
        }
        return regime.map(mapping).fillna(0.0)

    def _map_dex_regime(self, regime: pd.Series) -> pd.Series:
        """Map DEX regime to numeric score."""
        mapping = {
            "long_dex": 0.3,       # Dealers long = support
            "short_dex": -0.3,     # Dealers short = resistance
            "neutral_dex": 0.0
        }
        return regime.map(mapping).fillna(0.0)

    def _map_entropy_regime(self, regime: pd.Series) -> pd.Series:
        """Map entropy regime to numeric score."""
        mapping = {
            "low_entropy": 0.4,    # Organized, trending
            "high_entropy": -0.4,  # Disorganized, choppy
            "normal": 0.0
        }
        return regime.map(mapping).fillna(0.0)

    def _map_iv_rv_regime(self, regime: pd.Series) -> pd.Series:
        """Map IV/RV regime to numeric score."""
        mapping = {
            "iv_expensive": -0.2,  # Options expensive, sell vol
            "iv_cheap": 0.2,       # Options cheap, buy vol
            "neutral": 0.0
        }
        return regime.map(mapping).fillna(0.0)

    # Implement abstract method
    def classify(self, data: pd.DataFrame) -> RegimeResult:
        result = self.classify(data)
        return RegimeResult(
            regime=result.primary_regime,
            confidence=result.confidence,
            components=result.components
        )


def get_regime_summary(result: CombinedRegimeResult, timestamp: pd.Timestamp) -> dict:
    """Get human-readable regime summary at timestamp."""
    idx = result.primary_regime.index.searchsorted(timestamp) - 1
    if idx < 0:
        return {}

    return {
        "primary_regime": result.primary_regime.iloc[idx],
        "regime_score": result.regime_score.iloc[idx],
        "confidence": result.confidence.iloc[idx],
        "bias": result.bias.iloc[idx],
        "sub_regimes": {k: v.iloc[idx] for k, v in result.sub_regimes.items()},
        "components": {k: {k2: v2.iloc[idx] if hasattr(v2, 'iloc') else v2
                          for k2, v2 in v.items()} for k, v in result.components.items()}
    }