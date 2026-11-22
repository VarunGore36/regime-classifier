"""GARCH volatility modeling and Shadow Gamma features."""

from dataclasses import dataclass
from typing: Optional
import pandas as pd
import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm


@dataclass
class GARCHResult:
    """GARCH model results."""
    conditional_vol: pd.Series      # Conditional volatility (sigma_t)
    log_likelihood: float           # Log-likelihood
    params: dict                    # Estimated parameters
    residuals: pd.Series            # Standardized residuals
    forecast: pd.Series             # 1-step ahead forecast
    vol_regime: pd.Series           # Vol regime from GARCH


@dataclass
class ShadowGammaResult:
    """Shadow Gamma results."""
    shadow_gamma: pd.Series         # Shadow gamma estimate
    gamma_regime: pd.Series         # Gamma regime
    hedging_pressure: pd.Series     # Estimated dealer hedging pressure


def garch11_log_likelihood(params, returns):
    """Negative log-likelihood for GARCH(1,1)."""
    omega, alpha, beta = params

    if omega <= 0 or alpha < 0 or beta < 0 or alpha + beta >= 1:
        return 1e10

    T = len(returns)
    sigma2 = np.zeros(T)
    sigma2[0] = np.var(returns)

    for t in range(1, T):
        sigma2[t] = omega + alpha * returns[t-1]**2 + beta * sigma2[t-1]

    # Check for invalid variances
    if np.any(sigma2 <= 0):
        return 1e10

    loglik = -0.5 * T * np.log(2 * np.pi) - 0.5 * np.sum(np.log(sigma2) + returns**2 / sigma2)
    return -loglik


def estimate_garch11(
    returns: pd.Series,
    initial_guess: tuple = (0.01, 0.1, 0.85)
) -> tuple:
    """
    Estimate GARCH(1,1) parameters.

    Returns: (omega, alpha, beta)
    """
    # Use simple initial guess
    var = np.var(returns)
    omega_init = var * 0.01
    alpha_init = 0.1
    beta_init = 0.85

    result = minimize(
        garch11_log_likelihood,
        (omega_init, alpha_init, beta_init),
        args=(returns.values,),
        method='L-BFGS-B',
        bounds=[(1e-6, None), (1e-6, 1), (1e-6, 1)],
        options={'maxiter': 1000}
    )

    if not result.success:
        # Return reasonable defaults
        return omega_init, alpha_init, beta_init

    omega, alpha, beta = result.x
    # Enforce stationarity
    if alpha + beta >= 1:
        beta = 0.99 - alpha

    return omega, alpha, beta


def calculate_garch_volatility(
    returns: pd.Series,
    omega: float = None,
    alpha: float = None,
    beta: float = None,
    initial_var: float = None,
    reestimate_freq: int = 252 * 390  # Re-estimate annually
) -> GARCHResult:
    """
    Calculate GARCH(1,1) conditional volatility.

    sigma_t^2 = omega + alpha * r_{t-1}^2 + beta * sigma_{t-1}^2
    """
    rets = returns.dropna().values

    # Estimate parameters if not provided
    if omega is None or alpha is None or beta is None:
        omega, alpha, beta = estimate_garch11(returns)

    if initial_var is None:
        initial_var = np.var(rets)

    T = len(rets)
    sigma2 = np.zeros(T)
    sigma2[0] = initial_var

    # Rolling re-estimation
    params_history = []

    for t in range(1, T):
        # Re-estimate periodically
        if t % reestimate_freq == 0 and t > 1000:
            omega, alpha, beta = estimate_garch11(pd.Series(rets[:t]))
            params_history.append((t, omega, alpha, beta))

        sigma2[t] = omega + alpha * rets[t-1]**2 + beta * sigma2[t-1]

    conditional_vol = pd.Series(np.sqrt(sigma2), index=returns.dropna().index)
    conditional_vol.name = "garch_vol"

    # Standardized residuals
    residuals = returns.dropna() / conditional_vol

    # 1-step ahead forecast
    forecast = pd.Series(index=returns.dropna().index, dtype=float)
    for t in range(T):
        if t == 0:
            forecast.iloc[t] = np.sqrt(omega + alpha * rets[0]**2 + beta * initial_var)
        else:
            forecast.iloc[t] = np.sqrt(omega + alpha * rets[t-1]**2 + beta * sigma2[t-1])

    # Vol regime from GARCH
    vol_percentile = conditional_vol.rolling(252 * 390, min_periods=1000).rank(pct=True)
    vol_regime = pd.Series("normal", index=conditional_vol.index)
    vol_regime[vol_percentile > 0.8] = "high_vol"
    vol_regime[vol_percentile < 0.2] = "low_vol"

    params = {"omega": omega, "alpha": alpha, "beta": beta}
    if params_history:
        params["history"] = params_history

    # Log-likelihood
    loglik = -garch11_log_likelihood((omega, alpha, beta), rets)

    return GARCHResult(
        conditional_vol=conditional_vol,
        log_likelihood=loglik,
        params=params,
        residuals=residuals,
        forecast=forecast,
        vol_regime=vol_regime
    )


def calculate_egarch(
    returns: pd.Series,
    initial_params: tuple = None
) -> GARCHResult:
    """
    Estimate EGARCH(1,1) for asymmetric volatility.

    log(sigma_t^2) = omega + alpha * (|z_{t-1}| - E[|z|]) + gamma * z_{t-1} + beta * log(sigma_{t-1}^2)
    """
    # Simplified EGARCH - would need proper implementation
    return calculate_garch_volatility(returns)


def calculate_gjr_garch(
    returns: pd.Series,
    initial_params: tuple = None
) -> GARCHResult:
    """
    Estimate GJR-GARCH(1,1) for leverage effect.

    sigma_t^2 = omega + alpha * r_{t-1}^2 + gamma * r_{t-1}^2 * I(r_{t-1} < 0) + beta * sigma_{t-1}^2
    """
    # Simplified - would need proper implementation
    return calculate_garch_volatility(returns)


def calculate_shadow_gamma(
    options_chain: pd.DataFrame,
    underlying_price: pd.Series,
    spot_gamma_col: str = "gamma",
    oi_col: str = "oi",
    delta_col: str = "delta",
    right_col: str = "right",
    timestamp_col: str = "timestamp",
    expiry_col: str = "expiry",
    strike_col: str = "strike",
    underlying_col: str = "underlying_price"
) -> ShadowGammaResult:
    """
    Calculate Shadow Gamma (unobserved gamma from dark pools/OTC).

    Based on: Shadow Gamma = Total Gamma - Exchange Gamma
    Estimates hidden dealer positioning.

    This is a simplified proxy using:
    - OI changes at key strikes
    - Delta hedging imbalances
    - Volume/IV anomalies
    """
    results = []

    for ts in options_chain[timestamp_col].unique():
        chain = options_chain[options_chain[timestamp_col] == ts].copy()
        spot = underlying_price.get(ts, chain[underlying_col].iloc[0])

        # Total exchange gamma
        total_gamma = (chain[oi_col] * chain[spot_gamma_col] * 100 * chain[underlying_col]).sum()

        # Gamma by strike
        gamma_by_strike = chain.groupby(strike_col).apply(
            lambda x: (x[oi_col] * x[spot_gamma_col] * 100 * x[underlying_col]).sum()
        )

        # Gamma concentration (HHI)
        gamma_shares = gamma_by_strike / gamma_by_strike.sum()
        hhi = (gamma_shares ** 2).sum()

        # Delta imbalance
        delta_total = (chain[oi_col] * chain[delta_col] * 100 * chain[underlying_col]).sum()

        # OI change proxy (would need previous snapshot)
        # For now use volume/OI ratio as activity proxy
        vol_oi_ratio = chain["volume"].sum() / chain[oi_col].sum() if chain[oi_col].sum() > 0 else 0

        # Shadow gamma estimate (simplified)
        # High HHI + high delta imbalance + high vol/OI = potential shadow gamma
        shadow_gamma = total_gamma * hhi * (1 + abs(delta_total) / 1e9) * (1 + vol_oi_ratio)

        # Gamma regime
        if hhi > 0.3 and abs(delta_total) > 1e9:
            gamma_regime = "concentrated"
        elif hhi < 0.1:
            gamma_regime = "dispersed"
        else:
            gamma_regime = "normal"

        # Hedging pressure
        # Positive = dealers need to buy (short gamma)
        # Negative = dealers need to sell (long gamma)
        hedging_pressure = -delta_total / 1e9  # Normalized

        results.append({
            "timestamp": ts,
            "total_gamma": total_gamma,
            "shadow_gamma": shadow_gamma,
            "gamma_hhi": hhi,
            "delta_total": delta_total,
            "vol_oi_ratio": vol_oi_ratio,
            "gamma_regime": gamma_regime,
            "hedging_pressure": hedging_pressure
        })

    df = pd.DataFrame(results).set_index("timestamp")

    return ShadowGammaResult(
        shadow_gamma=df["shadow_gamma"],
        gamma_regime=df["gamma_regime"],
        hedging_pressure=df["hedging_pressure"]
    )


def get_garch_signals(
    garch_result: GARCHResult,
    returns: pd.Series,
    threshold: float = 2.0
) -> pd.DataFrame:
    """
    Generate signals from GARCH volatility.
    """
    df = pd.DataFrame(index=garch_result.conditional_vol.index)

    df["garch_vol"] = garch_result.conditional_vol
    df["garch_forecast"] = garch_result.forecast
    df["vol_regime"] = garch_result.vol_regime
    df["residuals"] = garch_result.residuals

    # Vol expansion/contraction
    df["vol_expanding"] = df["garch_vol"].diff() > 0
    df["vol_contracting"] = df["garch_vol"].diff() < 0

    # Extreme residuals (potential mean reversion)
    df["extreme_residual"] = df["residuals"].abs() > threshold

    # Vol regime signals
    df["high_vol_regime"] = df["vol_regime"] == "high_vol"
    df["low_vol_regime"] = df["vol_regime"] == "low_vol"

    return df


def get_shadow_gamma_signals(
    shadow_result: ShadowGammaResult
) -> pd.DataFrame:
    """
    Generate signals from Shadow Gamma.
    """
    df = pd.DataFrame(index=shadow_result.shadow_gamma.index)

    df["shadow_gamma"] = shadow_result.shadow_gamma
    df["gamma_regime"] = shadow_result.gamma_regime
    df["hedging_pressure"] = shadow_result.hedging_pressure

    # Regime signals
    df["gamma_concentrated"] = df["gamma_regime"] == "concentrated"
    df["gamma_dispersed"] = df["gamma_regime"] == "dispersed"

    # Hedging pressure signals
    df["dealers_buy"] = df["hedging_pressure"] > 1.0  # Dealers need to buy
    df["dealers_sell"] = df["hedging_pressure"] < -1.0  # Dealers need to sell

    return df