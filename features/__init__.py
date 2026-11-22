"""Features package."""

from .volume_profile import VolumeProfile, calculate_volume_profile, calculate_tpo_profile, get_profile_levels
from .vwap import VWAPResult, calculate_vwap, calculate_rolling_vwap, get_vwap_signals
from .realized_vol import RealizedVolResult, calculate_realized_volatility, calculate_rolling_volatility, calculate_parkinson_volatility, calculate_garman_klass_volatility, get_volatility_signals
from .implied_vol import IVResult, calculate_expected_move, calculate_iv_features, calculate_iv_rv_ratio, get_iv_signals
from .greeks import GreeksResult, black_scholes_greeks, black_scholes_price, implied_volatility, calculate_greeks_for_chain, black76_greeks, black76_price
from .conversion import ConversionResult, calculate_conversion_ratio, convert_qqq_to_nq, convert_nq_to_qqq, convert_key_levels, get_conversion_signals, dynamic_conversion_ratio
from .cvd import CVDResult, calculate_cvd, calculate_delta_profile, get_delta_signals
from .absorption import AbsorptionResult, ExhaustionResult, TrappedTradersResult, detect_absorption, detect_exhaustion, detect_trapped_traders, detect_delta_divergence, get_orderflow_signals
from .entropy import EntropyResult, calculate_shannon_entropy, calculate_permutation_entropy, calculate_sample_entropy, calculate_approximate_entropy, calculate_multiscale_entropy, classify_entropy_regime, get_entropy_features
from .iv_skew import IVSkewResult, IVTermStructureResult, calculate_iv_skew, calculate_term_structure, get_skew_signals
from .garch import GARCHResult, ShadowGammaResult, estimate_garch11, calculate_garch_volatility, calculate_egarch, calculate_gjr_garch, calculate_shadow_gamma, get_garch_signals, get_shadow_gamma_signals

__all__ = [
    # Volume Profile
    "VolumeProfile", "calculate_volume_profile", "calculate_tpo_profile", "get_profile_levels",
    # VWAP
    "VWAPResult", "calculate_vwap", "calculate_rolling_vwap", "get_vwap_signals",
    # Realized Vol
    "RealizedVolResult", "calculate_realized_volatility", "calculate_rolling_volatility",
    "calculate_parkinson_volatility", "calculate_garman_klass_volatility", "get_volatility_signals",
    # Implied Vol
    "IVResult", "calculate_expected_move", "calculate_iv_features", "calculate_iv_rv_ratio", "get_iv_signals",
    # Greeks
    "GreeksResult", "black_scholes_greeks", "black_scholes_price", "implied_volatility",
    "calculate_greeks_for_chain", "black76_greeks", "black76_price",
    # Conversion
    "ConversionResult", "calculate_conversion_ratio", "convert_qqq_to_nq", "convert_nq_to_qqq",
    "convert_key_levels", "get_conversion_signals", "dynamic_conversion_ratio",
    # CVD
    "CVDResult", "calculate_cvd", "calculate_delta_profile", "get_delta_signals",
    # Absorption/Exhaustion
    "AbsorptionResult", "ExhaustionResult", "TrappedTradersResult",
    "detect_absorption", "detect_exhaustion", "detect_trapped_traders",
    "detect_delta_divergence", "get_orderflow_signals",
    # Entropy
    "EntropyResult", "calculate_shannon_entropy", "calculate_permutation_entropy",
    "calculate_sample_entropy", "calculate_approximate_entropy", "calculate_multiscale_entropy",
    "classify_entropy_regime", "get_entropy_features",
    # IV Skew
    "IVSkewResult", "IVTermStructureResult", "calculate_iv_skew", "calculate_term_structure", "get_skew_signals",
    # GARCH
    "GARCHResult", "ShadowGammaResult", "estimate_garch11", "calculate_garch_volatility",
    "calculate_egarch", "calculate_gjr_garch", "calculate_shadow_gamma",
    "get_garch_signals", "get_shadow_gamma_signals",
]