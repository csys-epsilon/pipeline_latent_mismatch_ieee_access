"""
Memorization diagnostics for synthetic clinical data.

Implements the four nearest-neighbor diagnostics reported in Manuscript v2.3
Section IV-D and Table VII:

    1) 1-NN ratio        = mean(synthetic-to-train) / mean(real-test-to-train)
    2) DCR ratio         = median(synthetic-to-train) / median(real-test-to-train)
                           (Distance to Closest Record ratio)
    3) Hit rate at P5    = fraction of synthetic samples whose nearest-neighbor
                           distance to train falls within the 5th percentile of
                           real-test-to-train distances (×100, in %)
    4) NNDR              = median ratio of 1st-NN distance to 2nd-NN distance,
                           reported separately for synthetic and real-test

All four are computed in the standardized original feature space using L2 norm,
as specified in Section III-D. Target columns (is_mi, is_stroke) are excluded
from the feature space; missing values are zero-filled prior to standardization.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


def compute_nn_diagnostics(real_train, real_test, syn, exclude_cols=None):
    """
    Compute all four memorization diagnostics for one (train, test, syn) cell.

    Parameters
    ----------
    real_train : pd.DataFrame
        Real training cohort (e.g. K-1 folds concatenated).
    real_test : pd.DataFrame
        Real held-out fold.
    syn : pd.DataFrame
        Synthetic samples produced for this fold.
    exclude_cols : list[str] | None
        Columns to exclude from the feature space (defaults to target labels).

    Returns
    -------
    dict
        Keys: 1-NN_ratio, Syn_to_train_mean, Real_to_train_mean,
              DCR_syn, DCR_test, DCR_ratio, epsilon_P5,
              hit_rate_syn_pct, hit_rate_test_pct, NNDR_syn, NNDR_test.
    """
    if exclude_cols is None:
        exclude_cols = ['is_mi', 'is_stroke']

    common = [c for c in real_train.columns
              if c not in exclude_cols and c in syn.columns]
    if len(common) < 5:
        raise ValueError(
            f"Too few common feature columns for NN diagnostics: {len(common)}"
        )

    sc = StandardScaler()
    X_train = sc.fit_transform(real_train[common].fillna(0))
    X_test  = sc.transform(real_test[common].fillna(0))
    X_syn   = sc.transform(syn[common].fillna(0))

    nn = NearestNeighbors(n_neighbors=2).fit(X_train)
    dist_syn,  _ = nn.kneighbors(X_syn)
    dist_test, _ = nn.kneighbors(X_test)

    # 1) 1-NN ratio (means)
    syn_mean  = float(dist_syn[:, 0].mean())
    test_mean = float(dist_test[:, 0].mean())
    nn_ratio  = syn_mean / (test_mean + 1e-12)

    # 2) DCR ratio (medians)
    dcr_syn  = float(np.median(dist_syn[:, 0]))
    dcr_test = float(np.median(dist_test[:, 0]))
    dcr_ratio = dcr_syn / (dcr_test + 1e-12)

    # 3) Hit rate at P5 of real-test-to-train distances
    eps = float(np.percentile(dist_test[:, 0], 5))
    hit_syn  = float((dist_syn[:, 0]  <= eps).mean() * 100)
    hit_test = float((dist_test[:, 0] <= eps).mean() * 100)

    # 4) Nearest-neighbor distance ratio (NNDR; 1st-NN / 2nd-NN, sample medians)
    nndr_syn = float(np.median(
        dist_syn[:, 0] / np.maximum(dist_syn[:, 1], 1e-12)
    ))
    nndr_test = float(np.median(
        dist_test[:, 0] / np.maximum(dist_test[:, 1], 1e-12)
    ))

    return {
        '1-NN_ratio':         nn_ratio,
        'Syn_to_train_mean':  syn_mean,
        'Real_to_train_mean': test_mean,
        'DCR_syn':            dcr_syn,
        'DCR_test':           dcr_test,
        'DCR_ratio':          dcr_ratio,
        'epsilon_P5':         eps,
        'hit_rate_syn_pct':   hit_syn,
        'hit_rate_test_pct':  hit_test,
        'NNDR_syn':           nndr_syn,
        'NNDR_test':          nndr_test,
    }


def aggregate_arch_scaler(df_raw, metric_cols=None):
    """
    Pool the raw per-cell diagnostics into per-(architecture × scaler) summaries
    with 95% normal-approximation confidence intervals.

    Parameters
    ----------
    df_raw : pd.DataFrame
        One row per (arch × scaler × seed × fold) cell with the metric columns
        produced by compute_nn_diagnostics().
    metric_cols : list[str] | None
        Metrics to summarize. Defaults to the four headline metrics.

    Returns
    -------
    pd.DataFrame
    """
    if metric_cols is None:
        metric_cols = ['1-NN_ratio', 'DCR_ratio',
                       'hit_rate_syn_pct', 'NNDR_syn']

    rows = []
    for (arch, scaler), g in df_raw.groupby(['arch', 'scaler']):
        n = len(g)
        row = {'arch': arch, 'scaler': scaler, 'n_cells': n}
        for m in metric_cols:
            if m not in g.columns:
                continue
            mean = float(g[m].mean())
            sem = float(g[m].std() / np.sqrt(n)) if n > 1 else 0.0
            row[f'{m}_mean']    = round(mean, 4)
            row[f'{m}_CI_low']  = round(mean - 1.96 * sem, 4)
            row[f'{m}_CI_high'] = round(mean + 1.96 * sem, 4)
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate_arch_level(df_raw, metric_cols=None):
    """
    Pool across all scalers within each architecture (architecture-level summary).
    Reproduces the architecture comparison used in Section IV-D.
    """
    if metric_cols is None:
        metric_cols = ['1-NN_ratio', 'DCR_ratio',
                       'hit_rate_syn_pct', 'NNDR_syn']

    rows = []
    for arch, g in df_raw.groupby('arch'):
        n = len(g)
        row = {'arch': arch, 'n_cells': n}
        for m in metric_cols:
            if m not in g.columns:
                continue
            mean = float(g[m].mean())
            sem = float(g[m].std() / np.sqrt(n)) if n > 1 else 0.0
            row[f'{m}_mean']    = round(mean, 4)
            row[f'{m}_CI_low']  = round(mean - 1.96 * sem, 4)
            row[f'{m}_CI_high'] = round(mean + 1.96 * sem, 4)
        rows.append(row)
    return pd.DataFrame(rows)
