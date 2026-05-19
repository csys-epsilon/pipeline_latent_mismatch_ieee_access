"""
Component-level latent-space quality: Latent KL divergence.

Implements the Latent KL metric described in Manuscript v2.3 Section III-D:

    "Latent Kullback–Leibler (KL) divergence is computed between each
     scaler's transformed output and the standard normal distribution,
     where lower values indicate closer alignment to the latent prior.
     Estimation uses kernel density over a 500-point grid spanning the
     empirical support, evaluated across five scalers and seven HK variables."

Returns one KL value per (scaler × variable) cell. Aggregating across the
seven HK variables gives the HK-mean Latent KL plotted on the x-axis of
Fig. 3A.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


# numpy >= 2.0 renamed np.trapz to np.trapezoid; keep both code paths working.
_trapz = getattr(np, 'trapezoid', getattr(np, 'trapz', None))


def _kde_pdf(x, grid):
    """Gaussian KDE evaluated on `grid`. Returns a normalized density."""
    if len(x) < 2 or np.std(x) < 1e-12:
        # Degenerate case: return a near-uniform tiny density to avoid 0/0
        return np.full_like(grid, 1e-10, dtype=float)
    kde = stats.gaussian_kde(x)
    pdf = kde.evaluate(grid)
    pdf = np.maximum(pdf, 1e-12)
    return pdf


def latent_kl(scaled_values, n_grid=500):
    """
    KL( p_data || N(0, 1) ) on a 500-point empirical-support grid.

    Parameters
    ----------
    scaled_values : 1-D array-like of scaler-transformed values
    n_grid        : number of grid points (default 500, matching manuscript)

    Returns
    -------
    kl : float, KL divergence in nats
    """
    x = np.asarray(scaled_values, dtype=float).ravel()
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return float('nan')

    lo, hi = float(np.min(x)), float(np.max(x))
    pad = max(1e-3, 0.05 * (hi - lo))
    grid = np.linspace(lo - pad, hi + pad, n_grid)

    p = _kde_pdf(x, grid)
    p /= _trapz(p, grid)

    q = stats.norm.pdf(grid)
    q = np.maximum(q, 1e-12)
    q /= _trapz(q, grid)

    return float(_trapz(p * np.log(p / q), grid))


def latent_kl_table(scaled_dataframe, hk_features):
    """
    Compute Latent KL for each HK variable in a scaler-transformed dataframe.

    Returns
    -------
    pd.DataFrame with columns ['Feature', 'LatentKL']
    """
    rows = []
    for col in hk_features:
        if col not in scaled_dataframe.columns:
            continue
        rows.append({
            'Feature':  col,
            'LatentKL': latent_kl(scaled_dataframe[col].values),
        })
    return pd.DataFrame(rows)
