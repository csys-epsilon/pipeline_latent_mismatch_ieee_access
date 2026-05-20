"""
Component-level latent-space quality: Latent KL divergence.

Implements the Latent KL metric described in Manuscript v2.3 Section III-D:

    "Latent Kullback-Leibler (KL) divergence is computed between each
     scaler's transformed output and the standard normal distribution,
     where lower values indicate closer alignment to the latent prior.
     Estimation uses kernel density over a grid that adaptively spans
     the empirical support while guaranteeing standard-normal tail
     coverage, evaluated across five scalers and seven HK variables."

The estimator computes

    KL(p || q) = integral p(x) * log(p(x) / q(x)) dx

where p is a Gaussian KDE of the scaler-transformed values and q is the
standard normal density N(0, 1).

Returns one KL value per (scaler x variable) cell. Aggregating across the
seven HK variables gives the HK-mean Latent KL plotted on the x-axis of
Fig. 3A.

Implementation notes
--------------------
1. Continuous KL via trapezoidal integration. The integrand p * log(p/q)
   is integrated explicitly with dx, not discretized via sum-normalization
   of pdf values. This avoids the silent omission of dx that would
   otherwise distort the integral when the grid is non-uniform or extends
   beyond the bulk of mass.

2. Dynamic grid range. The grid spans
       [-max(MIN_HALFWIDTH, |z|_max * PAD), +max(...)]
   so that (a) outlier tails of shape-preserving scalers (Standard,
   Robust) are not truncated, and (b) the standard normal tail mass is
   still captured (MIN_HALFWIDTH = 8 -> 1 - 2 * Phi(-8) > 1 - 1e-15).
   The pdf values are used directly; no post-hoc renormalization that
   would distort the densities by clipping support is applied.

3. Bandwidth. Silverman's rule of thumb is used explicitly for
   reproducibility. Heavy-tail distributions are slightly over-smoothed
   under this rule; results should be interpreted as an upper-bound
   surrogate for tail-region density mismatch rather than a tight
   density estimate.

4. Degenerate cases. Constant inputs (std < 1e-12) and inputs with
   fewer than two finite samples return NaN, matching the manuscript's
   "no estimate" convention.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


# numpy >= 2.0 renamed np.trapz to np.trapezoid; keep both code paths working.
_trapz = getattr(np, 'trapezoid', getattr(np, 'trapz', None))


# Estimator parameters (match Fig. 3 production code exactly).
KDE_GRID_N = 2000             # grid density for trapezoidal integration
KDE_GRID_PAD = 1.5            # padding multiplier beyond observed |z|.max()
KDE_GRID_MIN_HALFWIDTH = 8.0  # minimum |x| extent to cover N(0,1) tail mass
KDE_BW_METHOD = 'silverman'   # KDE bandwidth rule; see note 3 above
PDF_FLOOR = 1e-12             # numerical floor for log(p), log(q)


def _make_grid(z, n=KDE_GRID_N,
               pad=KDE_GRID_PAD,
               min_halfwidth=KDE_GRID_MIN_HALFWIDTH):
    """Build a symmetric grid that covers both the empirical z range
    (with `pad` multiplier) and the standard normal tail
    (at least +/- min_halfwidth).
    """
    z_abs_max = float(np.abs(z).max())
    half = max(min_halfwidth, z_abs_max * pad)
    return np.linspace(-half, half, n)


def latent_kl(scaled_values, bw_method=KDE_BW_METHOD,
              n_grid=KDE_GRID_N,
              min_halfwidth=KDE_GRID_MIN_HALFWIDTH):
    """
    KL( p_data || N(0, 1) ) via Gaussian KDE on a tail-safe symmetric grid.

    Parameters
    ----------
    scaled_values : 1-D array-like of scaler-transformed values
    bw_method     : KDE bandwidth rule. Default 'silverman'. Accepts the
                    same values as scipy.stats.gaussian_kde.
    n_grid        : number of grid points for trapezoidal integration.
                    Default 2000.
    min_halfwidth : minimum half-width of the grid. Default 8.0 ensures
                    the standard normal q has its tail mass fully captured
                    (Phi(-8) < 1e-15).

    Returns
    -------
    kl : float, KL divergence in nats. NaN for degenerate inputs.
    """
    x = np.asarray(scaled_values, dtype=float).ravel()
    x = x[np.isfinite(x)]
    if len(x) < 2 or np.std(x) < 1e-12:
        return float('nan')

    try:
        kde = stats.gaussian_kde(x, bw_method=bw_method)
    except Exception:
        return float('nan')

    grid = _make_grid(x, n=n_grid, min_halfwidth=min_halfwidth)

    p = kde.evaluate(grid)
    q = stats.norm.pdf(grid, loc=0.0, scale=1.0)

    p = np.maximum(p, PDF_FLOOR)
    q = np.maximum(q, PDF_FLOOR)

    # Proper continuous KL via trapezoidal rule.
    integrand = p * (np.log(p) - np.log(q))
    return float(_trapz(integrand, grid))


def latent_kl_table(scaled_dataframe, hk_features,
                    bw_method=KDE_BW_METHOD,
                    n_grid=KDE_GRID_N,
                    min_halfwidth=KDE_GRID_MIN_HALFWIDTH):
    """
    Compute Latent KL for each HK variable in a scaler-transformed dataframe.

    Parameters
    ----------
    scaled_dataframe : pandas.DataFrame with one column per variable, values
                       already produced by the scaler under evaluation.
    hk_features      : iterable of column names corresponding to the
                       high-kurtosis variables to evaluate.
    bw_method, n_grid, min_halfwidth :
                       forwarded to `latent_kl`. See that function's
                       docstring for defaults and meaning.

    Returns
    -------
    pd.DataFrame with columns ['Feature', 'LatentKL']. Columns not present
    in `scaled_dataframe` are skipped silently.
    """
    rows = []
    for col in hk_features:
        if col not in scaled_dataframe.columns:
            continue
        rows.append({
            'Feature':  col,
            'LatentKL': latent_kl(
                scaled_dataframe[col].values,
                bw_method=bw_method,
                n_grid=n_grid,
                min_halfwidth=min_halfwidth,
            ),
        })
    return pd.DataFrame(rows)