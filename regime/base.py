"""Base regime classifier."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing: Optional
import pandas as pd
import numpy as np


@dataclass
class RegimeResult:
    """Regime classification result."""
    regime: pd.Series                    # Regime label per bar
    confidence: pd.Series                # Confidence score (0-1)
    components: dict                     # Component scores


class RegimeClassifier(ABC):
    """Abstract base class for regime classifiers."""

    @abstractmethod
    def classify(self, data: pd.DataFrame) -> RegimeResult:
        """Classify market regime."""
        pass

    def get_regime_at(self, result: RegimeResult, timestamp: pd.Timestamp) -> str:
        """Get regime at specific timestamp."""
        if timestamp in result.regime.index:
            return result.regime.loc[timestamp]
        # Find closest previous
        idx = result.regime.index.searchsorted(timestamp) - 1
        if idx >= 0:
            return result.regime.iloc[idx]
        return "unknown"


class CompositeRegimeClassifier(RegimeClassifier):
    """Combine multiple regime classifiers with weights."""

    def __init__(self, classifiers: dict, weights: Optional[dict] = None):
        """
        Args:
            classifiers: Dict of {name: RegimeClassifier}
            weights: Dict of {name: weight} (normalized to sum=1)
        """
        self.classifiers = classifiers
        if weights is None:
            weights = {name: 1.0 for name in classifiers}
        total = sum(weights.values())
        self.weights = {k: v / total for k, v in weights.items()}

    def classify(self, data: pd.DataFrame) -> RegimeResult:
        results = {}
        for name, clf in self.classifiers.items():
            results[name] = clf.classify(data)

        # Combine regimes by weighted voting
        regime_votes = pd.DataFrame({name: r.regime for name, r in results.items()})
        confidence_scores = pd.DataFrame({name: r.confidence for name, r in results.items()})

        # Weighted vote
        combined_confidence = pd.Series(0.0, index=data.index)
        for name, weight in self.weights.items():
            combined_confidence += confidence_scores[name] * weight

        # Majority vote for regime (simplified: use highest confidence classifier)
        primary_clf = max(self.weights, key=self.weights.get)
        combined_regime = results[primary_clf].regime

        # Components
        components = {name: r.components for name, r in results.items()}

        return RegimeResult(
            regime=combined_regime,
            confidence=combined_confidence,
            components=components
        )