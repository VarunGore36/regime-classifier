"""Entropy-based market regime features."""

from dataclasses import dataclass
from typing: Optional
import pandas as pd
import numpy as np
from scipy.stats import entropy as scipy_entropy


@dataclass
class EntropyResult:
    """Entropy calculation results."""
    shannon_entropy: pd.Series          # Shannon entropy
    permutation_entropy: pd.Series      # Permutation entropy
    sample_entropy: pd.Series           # Sample entropy
    approximate_entropy: pd.Series      # Approximate entropy
    entropy_regime: pd.Series           # Regime classification


def calculate_shannon_entropy(
    returns: pd.Series,
    window: int = 100,
    bins: int = 10
) -> pd.Series:
    """
    Calculate Shannon entropy of return distribution.

    H = -sum(p * log(p)) where p is probability of each bin.
    """
    ent = pd.Series(index=returns.index, dtype=float)

    for i in range(window, len(returns)):
        window_returns = returns.iloc[i-window:i].dropna()
        if len(window_returns) < 10:
            ent.iloc[i] = np.nan
            continue

        hist, _ = np.histogram(window_returns, bins=bins, density=True)
        hist = hist[hist > 0]
        ent.iloc[i] = -np.sum(hist * np.log(hist))

    return ent


def calculate_permutation_entropy(
    returns: pd.Series,
    window: int = 100,
    order: int = 3,
    delay: int = 1
) -> pd.Series:
    """
    Calculate permutation entropy.

    Based on Bandt & Pompe (2002). Measures complexity of time series.
    """
    from itertools import permutations

    # All possible permutations of given order
    perms = list(permutations(range(order)))
    perm_to_idx = {p: i for i, p in enumerate(perms)}

    ent = pd.Series(index=returns.index, dtype=float)

    for i in range(window, len(returns)):
        window_returns = returns.iloc[i-window:i].dropna()
        if len(window_returns) < order * delay + 1:
            ent.iloc[i] = np.nan
            continue

        # Embedding
        embedded = []
        for j in range(len(window_returns) - (order - 1) * delay):
            pattern = tuple(window_returns.iloc[j + k * delay] for k in range(order))
            # Rank pattern
            ranked = tuple(np.argsort(pattern))
            embedded.append(ranked)

        # Count permutations
        counts = np.zeros(len(perms))
        for p in embedded:
            counts[perm_to_idx[p]] += 1

        # Normalize
        probs = counts / counts.sum()
        probs = probs[probs > 0]

        # Entropy
        ent.iloc[i] = -np.sum(probs * np.log(probs)) / np.log(len(perms))  # Normalized

    return ent


def calculate_sample_entropy(
    returns: pd.Series,
    window: int = 100,
    m: int = 2,
    r: float = 0.2
) -> pd.Series:
    """
    Calculate sample entropy (SampEn).

    Richman & Moorman (2000). Negative log likelihood that similar sequences remain similar.
    """
    ent = pd.Series(index=returns.index, dtype=float)

    for i in range(window, len(returns)):
        window_returns = returns.iloc[i-window:i].dropna().values
        if len(window_returns) < m + 1:
            ent.iloc[i] = np.nan
            continue

        # Standardize
        window_returns = (window_returns - window_returns.mean()) / window_returns.std()
        tolerance = r * window_returns.std()

        # Count matches for length m and m+1
        def count_matches(length):
            count = 0
            for j in range(len(window_returns) - length):
                template = window_returns[j:j+length]
                for k in range(j+1, len(window_returns) - length + 1):
                    if np.max(np.abs(template - window_returns[k:k+length])) <= tolerance:
                        count += 1
            return count

        A = count_matches(m + 1)
        B = count_matches(m)

        if B > 0 and A > 0:
            ent.iloc[i] = -np.log(A / B)
        else:
            ent.iloc[i] = np.nan

    return ent


def calculate_approximate_entropy(
    returns: pd.Series,
    window: int = 100,
    m: int = 2,
    r: float = 0.2
) -> pd.Series:
    """
    Calculate approximate entropy (ApEn).

    Pincus (1991). Similar to SampEn but includes self-matches.
    """
    ent = pd.Series(index=returns.index, dtype=float)

    for i in range(window, len(returns)):
        window_returns = returns.iloc[i-window:i].dropna().values
        if len(window_returns) < m + 1:
            ent.iloc[i] = np.nan
            continue

        # Standardize
        window_returns = (window_returns - window_returns.mean()) / window_returns.std()
        tolerance = r * window_returns.std()

        def phi(length):
            count = 0
            for j in range(len(window_returns) - length + 1):
                template = window_returns[j:j+length]
                matches = 0
                for k in range(len(window_returns) - length + 1):
                    if np.max(np.abs(template - window_returns[k:k+length])) <= tolerance:
                        matches += 1
                if matches > 0:
                    count += np.log(matches / (len(window_returns) - length + 1))
            return count / (len(window_returns) - length + 1)

        phi_m = phi(m)
        phi_m1 = phi(m + 1)
        ent.iloc[i] = phi_m - phi_m1

    return ent


def calculate_multiscale_entropy(
    returns: pd.Series,
    window: int = 200,
    max_scale: int = 10,
    m: int = 2,
    r: float = 0.2
) -> pd.DataFrame:
    """
    Calculate multiscale sample entropy.

    Costa et al. (2005). Entropy at multiple time scales.
    """
    results = {}

    for scale in range(1, max_scale + 1):
        # Coarse-grain
        if scale == 1:
            coarse = returns
        else:
            n = len(returns) // scale
            coarse = returns.iloc[:n*scale].values.reshape(n, scale).mean(axis=1)
            coarse = pd.Series(coarse, index=returns.iloc[scale-1::scale].index[:n])

        ent = calculate_sample_entropy(coarse, window=window//scale, m=m, r=r)
        results[f"scale_{scale}"] = ent

    return pd.DataFrame(results, index=returns.index)


def classify_entropy_regime(
    entropy: pd.Series,
    lookback: int = 252,
    low_threshold: float = 0.3,
    high_threshold: float = 0.7
) -> pd.Series:
    """
    Classify entropy regime.

    Low entropy = organized, trending
    High entropy = disorganized, choppy
    """
    # Normalize entropy to 0-1 range over lookback
    entropy_min = entropy.rolling(lookback).min()
    entropy_max = entropy.rolling(lookback).max()
    entropy_norm = (entropy - entropy_min) / (entropy_max - entropy_min)

    regime = pd.Series("normal", index=entropy.index)
    regime[entropy_norm < low_threshold] = "low_entropy"
    regime[entropy_norm > high_threshold] = "high_entropy"

    return regime


def get_entropy_features(
    returns: pd.Series,
    window: int = 100
) -> EntropyResult:
    """
    Calculate all entropy features.

    Returns EntropyResult with all entropy measures.
    """
    shannon = calculate_shannon_entropy(returns, window)
    perm = calculate_permutation_entropy(returns, window)
    sample = calculate_sample_entropy(returns, window)
    approx = calculate_approximate_entropy(returns, window)

    # Combine into regime
    # Use permutation entropy as primary (most robust)
    regime = classify_entropy_regime(perm, window * 2)

    return EntropyResult(
        shannon_entropy=shannon,
        permutation_entropy=perm,
        sample_entropy=sample,
        approximate_entropy=approx,
        entropy_regime=regime
    )