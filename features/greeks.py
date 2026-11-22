"""Black-Scholes Greeks calculation."""

from dataclasses import dataclass
from typing import Literal, Optional
import pandas as pd
import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq

from config import get_config


@dataclass
class GreeksResult:
    """Greeks calculation results."""
    delta: pd.Series
    gamma: pd.Series
    theta: pd.Series
    vega: pd.Series
    rho: pd.Series
    charm: pd.Series        # dDelta/dTime
    vanna: pd.Series        # dDelta/dVol
    vomma: pd.Series        # dVega/dVol


def black_scholes_greeks(
    S: float,               # Underlying price
    K: float,               # Strike price
    T: float,               # Time to expiration (years)
    r: float,               # Risk-free rate
    sigma: float,           # Implied volatility
    q: float = 0.0,         # Dividend yield
    option_type: Literal["call", "put"] = "call"
) -> dict:
    """
    Calculate all Black-Scholes Greeks.

    Returns dict with: delta, gamma, theta, vega, rho, charm, vanna, vomma
    """
    if T <= 0 or sigma <= 0:
        return {
            "delta": 0.0, "gamma": 0.0, "theta": 0.0, "vega": 0.0,
            "rho": 0.0, "charm": 0.0, "vanna": 0.0, "vomma": 0.0
        }

    sqrt_T = np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T

    # Standard normal PDF and CDF
    phi_d1 = norm.pdf(d1)
    Phi_d1 = norm.cdf(d1)
    Phi_d2 = norm.cdf(d2)

    if option_type == "call":
        delta = np.exp(-q * T) * Phi_d1
        theta = (-S * sigma * np.exp(-q * T) * phi_d1 / (2 * sqrt_T)
                 - r * K * np.exp(-r * T) * Phi_d2
                 + q * S * np.exp(-q * T) * Phi_d1) / 365
        rho = K * T * np.exp(-r * T) * Phi_d2 / 100
    else:
        delta = np.exp(-q * T) * (Phi_d1 - 1)
        theta = (-S * sigma * np.exp(-q * T) * phi_d1 / (2 * sqrt_T)
                 + r * K * np.exp(-r * T) * (1 - Phi_d2)
                 - q * S * np.exp(-q * T) * (1 - Phi_d1)) / 365
        rho = -K * T * np.exp(-r * T) * (1 - Phi_d2) / 100

    # Shared Greeks
    gamma = np.exp(-q * T) * phi_d1 / (S * sigma * sqrt_T)
    vega = S * np.exp(-q * T) * sqrt_T * phi_d1 / 100

    # Second-order Greeks
    # Charm = dDelta/dT
    charm = -np.exp(-q * T) * (phi_d1 * (r - q + 0.5 * sigma ** 2) / (sigma * sqrt_T)
                                + q * Phi_d1)

    # Vanna = dDelta/dSigma = dVega/dS
    vanna = np.exp(-q * T) * phi_d1 * d2 / sigma

    # Vomma = dVega/dSigma
    vomma = vega * d1 * d2 / sigma

    return {
        "delta": delta,
        "gamma": gamma,
        "theta": theta,
        "vega": vega,
        "rho": rho,
        "charm": charm,
        "vanna": vanna,
        "vomma": vomma
    }


def black_scholes_price(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    q: float = 0.0,
    option_type: Literal["call", "put"] = "call"
) -> float:
    """Calculate Black-Scholes option price."""
    if T <= 0:
        if option_type == "call":
            return max(S - K, 0)
        else:
            return max(K - S, 0)

    sqrt_T = np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T

    if option_type == "call":
        price = S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)

    return price


def implied_volatility(
    price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    q: float = 0.0,
    option_type: Literal["call", "put"] = "call",
    max_iter: int = 100,
    tol: float = 1e-6
) -> float:
    """
    Calculate implied volatility using Brent's method.
    """
    if T <= 0:
        return 0.0

    def objective(sigma):
        return black_scholes_price(S, K, T, r, sigma, q, option_type) - price

    # Check bounds
    try:
        iv = brentq(objective, 1e-6, 5.0, xtol=tol, maxiter=max_iter)
        return iv
    except ValueError:
        return np.nan


def calculate_greeks_for_chain(
    chain: pd.DataFrame,
    r: Optional[float] = None,
    q: Optional[float] = None,
    model: str = "black_scholes"
) -> pd.DataFrame:
    """
    Calculate Greeks for entire options chain.

    Args:
        chain: DataFrame with columns [underlying_price, strike, dte, iv, right]
        r: Risk-free rate (default from config)
        q: Dividend yield (default from config)
        model: "black_scholes" or "black76" (for futures options)

    Returns:
        DataFrame with Greeks columns added
    """
    config = get_config()

    if r is None:
        r = config.black_scholes.risk_free_rate
    if q is None:
        q = config.black_scholes.dividend_yield

    df = chain.copy()

    # Convert DTE to years
    df["T"] = df["dte"] / 365.0

    # Calculate Greeks row by row
    greeks_list = []
    for _, row in df.iterrows():
        greeks = black_scholes_greeks(
            S=row["underlying_price"],
            K=row["strike"],
            T=row["T"],
            r=r,
            sigma=row["iv"],
            q=q,
            option_type="call" if row["right"] == "C" else "put"
        )
        greeks_list.append(greeks)

    greeks_df = pd.DataFrame(greeks_list, index=df.index)
    return pd.concat([df, greeks_df], axis=1)


def black76_greeks(
    F: float,               # Futures price
    K: float,               # Strike price
    T: float,               # Time to expiration (years)
    r: float,               # Risk-free rate
    sigma: float,           # Implied volatility
    option_type: Literal["call", "put"] = "call"
) -> dict:
    """
    Black-76 Greeks for futures options.
    Uses futures price F instead of spot S.
    """
    if T <= 0 or sigma <= 0:
        return {
            "delta": 0.0, "gamma": 0.0, "theta": 0.0, "vega": 0.0,
            "rho": 0.0, "charm": 0.0, "vanna": 0.0, "vomma": 0.0
        }

    sqrt_T = np.sqrt(T)
    d1 = (np.log(F / K) + 0.5 * sigma ** 2 * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T

    phi_d1 = norm.pdf(d1)
    Phi_d1 = norm.cdf(d1)
    Phi_d2 = norm.cdf(d2)

    discount = np.exp(-r * T)

    if option_type == "call":
        delta = discount * Phi_d1
        theta = (-F * sigma * discount * phi_d1 / (2 * sqrt_T)
                 - r * K * discount * Phi_d2) / 365
        rho = -T * discount * (F * Phi_d1 - K * Phi_d2) / 100
    else:
        delta = discount * (Phi_d1 - 1)
        theta = (-F * sigma * discount * phi_d1 / (2 * sqrt_T)
                 + r * K * discount * (1 - Phi_d2)) / 365
        rho = T * discount * (F * (1 - Phi_d1) - K * (1 - Phi_d2)) / 100

    gamma = discount * phi_d1 / (F * sigma * sqrt_T)
    vega = F * discount * sqrt_T * phi_d1 / 100

    charm = -discount * (phi_d1 * (0.5 * sigma ** 2) / (sigma * sqrt_T))
    vanna = discount * phi_d1 * d2 / sigma
    vomma = vega * d1 * d2 / sigma

    return {
        "delta": delta,
        "gamma": gamma,
        "theta": theta,
        "vega": vega,
        "rho": rho,
        "charm": charm,
        "vanna": vanna,
        "vomma": vomma
    }


def black76_price(
    F: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: Literal["call", "put"] = "call"
) -> float:
    """Black-76 option price for futures."""
    if T <= 0:
        if option_type == "call":
            return max(F - K, 0)
        else:
            return max(K - F, 0)

    sqrt_T = np.sqrt(T)
    d1 = (np.log(F / K) + 0.5 * sigma ** 2 * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    discount = np.exp(-r * T)

    if option_type == "call":
        return discount * (F * norm.cdf(d1) - K * norm.cdf(d2))
    else:
        return discount * (K * norm.cdf(-d2) - F * norm.cdf(-d1))