"""
Linear mixed-effects analysis of kurtosis preservation across scalers.

Implements the model used in Manuscript v2.3 Section III-D and reported in
Table VIII (NHANES 18-HK Adaptive-centric pairwise contrasts):

    response   : kurtosis preservation rate (pp)
    fixed      : log(TTR), scaler (categorical; Quantile is the reference)
    random     : variable-level intercept
    estimation : restricted maximum likelihood (REML)
    overall    : likelihood-ratio test against the null, computed on ML refits
                 with df = 4 (one for each non-reference scaler)
    pairwise   : 10 Wald contrasts with Holm-corrected p-values

A variable-level TTR (P99/P50) is required as a fixed effect; supply it via
an external CSV with columns ['Feature', 'TTR'] when the long-format input
does not already include it.

Requires `statsmodels >= 0.14`.
"""

from __future__ import annotations

import os
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats

try:
    import statsmodels.formula.api as smf
    from statsmodels.stats.multitest import multipletests
except ImportError as exc:                                       # pragma: no cover
    raise ImportError(
        "statsmodels is required. Install with: pip install statsmodels"
    ) from exc


# ----------------------------------------------------------------------- #
# Input preparation
# ----------------------------------------------------------------------- #
def build_long_format(scaler_csv_paths, hk_variables=None,
                      kurt_real_col='Kurtosis_Real_Conv_Mean',
                      kurt_syn_col='Kurtosis_Syn_Conv_Mean',
                      scaler_label_map=None):
    """
    Concatenate per-scaler pooled univariate CSVs into one long-format frame.

    Parameters
    ----------
    scaler_csv_paths : dict
        Mapping {scaler_label: path/to/<scaler>_univariate_pooled_final.csv}.
        Example:
            {'Standard': '.../Standard_univariate_pooled_final.csv',
             'Quantile': '.../Quantile_univariate_pooled_final.csv', ...}
    hk_variables : list[str] | None
        Restrict to these features. If None, keeps every row.
    kurt_real_col, kurt_syn_col : str
        Column names for the fold-pooled mean real/synthetic kurtosis.
    scaler_label_map : dict | None
        Optional remapping for the 'scaler' column values (e.g. canonical case).

    Returns
    -------
    pd.DataFrame with columns ['Feature', 'scaler', kurt_real_col,
                               kurt_syn_col, 'preservation_pp']
    """
    frames = []
    for scaler, path in scaler_csv_paths.items():
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        df = pd.read_csv(path)
        for col in ('Feature', kurt_real_col, kurt_syn_col):
            if col not in df.columns:
                raise ValueError(
                    f"{path}: missing column {col!r}"
                )
        sub = df[['Feature', kurt_real_col, kurt_syn_col]].copy()
        sub['scaler'] = scaler if scaler_label_map is None \
                                else scaler_label_map.get(scaler, scaler)
        frames.append(sub)
    long_df = pd.concat(frames, ignore_index=True)

    long_df['preservation_pp'] = (
        long_df[kurt_syn_col] / long_df[kurt_real_col].replace(0, np.nan)
    ) * 100.0

    if hk_variables is not None:
        long_df = long_df[long_df['Feature'].isin(hk_variables)].copy()

    return long_df


def attach_ttr(long_df, ttr_csv_path,
               feature_col='Feature', ttr_col='TTR'):
    """
    Join a per-variable TTR column from an external CSV. Required columns:
    ['variable' or 'Feature', 'ttr' or 'TTR']. The function tolerates either
    naming convention and adds 'log_TTR' = log(TTR).
    """
    if not os.path.exists(ttr_csv_path):
        raise FileNotFoundError(ttr_csv_path)
    ttr_df = pd.read_csv(ttr_csv_path)

    rename_map = {}
    if 'variable' in ttr_df.columns and feature_col not in ttr_df.columns:
        rename_map['variable'] = feature_col
    if 'ttr' in ttr_df.columns and ttr_col not in ttr_df.columns:
        rename_map['ttr'] = ttr_col
    if rename_map:
        ttr_df = ttr_df.rename(columns=rename_map)

    if feature_col not in ttr_df.columns or ttr_col not in ttr_df.columns:
        raise ValueError(
            f"TTR CSV must contain {feature_col!r} and {ttr_col!r} columns"
        )

    merged = long_df.merge(
        ttr_df[[feature_col, ttr_col]], on=feature_col, how='left'
    )
    if merged[ttr_col].isna().any():
        miss = merged.loc[merged[ttr_col].isna(), feature_col].unique()
        raise ValueError(f"TTR missing for features: {sorted(miss)}")

    merged['log_TTR'] = np.log(merged[ttr_col])
    return merged


# ----------------------------------------------------------------------- #
# Model fitting
# ----------------------------------------------------------------------- #
def fit_mixedlm(df, response='preservation_pp', predictor='log_TTR',
                scaler_col='scaler', variable_col='Feature',
                reference_scaler='Quantile'):
    """
    Fit the manuscript's mixed-effects model and return summary objects.

    Returns
    -------
    md_full_reml : REML fit (used for fixed-effect inference and ICC)
    lrt_chi2     : LRT statistic (computed on ML refits)
    lrt_df       : LRT degrees of freedom (= number of non-reference scalers)
    lrt_pvalue   : LRT p-value from chi-square reference distribution
    icc          : ICC = var_random / (var_random + var_residual)
    var_random,
    var_residual : variance components from the REML fit
    """
    work = df.copy()

    # Ensure scaler is categorical with `reference_scaler` first
    levels = [reference_scaler] + sorted(
        s for s in work[scaler_col].unique() if s != reference_scaler
    )
    work[scaler_col] = pd.Categorical(work[scaler_col], categories=levels)

    formula_full = (
        f"{response} ~ {predictor} "
        f"+ C({scaler_col}, Treatment(reference='{reference_scaler}'))"
    )
    formula_null = f"{response} ~ {predictor}"

    md_full_reml = smf.mixedlm(formula_full, work, groups=work[variable_col]).fit(reml=True)

    var_random = float(md_full_reml.cov_re.iloc[0, 0])
    var_resid  = float(md_full_reml.scale)
    icc = var_random / (var_random + var_resid)

    # LRT must use ML, not REML
    md_full_ml = smf.mixedlm(formula_full, work, groups=work[variable_col]).fit(reml=False)
    md_null_ml = smf.mixedlm(formula_null, work, groups=work[variable_col]).fit(reml=False)
    lrt_chi2 = float(2.0 * (md_full_ml.llf - md_null_ml.llf))
    lrt_df = len(levels) - 1
    lrt_pvalue = float(1.0 - stats.chi2.cdf(lrt_chi2, df=lrt_df))

    return md_full_reml, lrt_chi2, lrt_df, lrt_pvalue, icc, var_random, var_resid


# ----------------------------------------------------------------------- #
# Pairwise contrasts (10 pairs, Holm-corrected)
# ----------------------------------------------------------------------- #
def pairwise_contrasts(md_full, scalers, reference_scaler='Quantile'):
    """
    Compute all C(|scalers|, 2) pairwise scaler contrasts with 95% Wald CIs
    and Holm-corrected p-values. For 5 scalers this yields 10 contrasts.
    """
    pairs = list(combinations(scalers, 2))
    params = md_full.params
    cov = md_full.cov_params()
    n_params = len(params)
    name_to_idx = {name: i for i, name in enumerate(params.index)}

    def _coef_name(scaler):
        if scaler == reference_scaler:
            return None
        return f"C(scaler, Treatment(reference='{reference_scaler}'))[T.{scaler}]"

    rows = []
    for s1, s2 in pairs:
        vec = np.zeros(n_params)
        est = 0.0
        c1 = _coef_name(s1)
        c2 = _coef_name(s2)
        if c1 is not None:
            vec[name_to_idx[c1]] += 1.0
            est += float(params[c1])
        if c2 is not None:
            vec[name_to_idx[c2]] -= 1.0
            est -= float(params[c2])

        var = float(vec @ cov.values @ vec)
        se = float(np.sqrt(max(var, 0.0)))
        if se > 0:
            z = est / se
            ci_low  = est - 1.96 * se
            ci_high = est + 1.96 * se
            p_raw = float(2.0 * (1.0 - stats.norm.cdf(abs(z))))
        else:
            z, ci_low, ci_high, p_raw = 0.0, est, est, 1.0

        rows.append({
            'Contrast': f"{s1} - {s2}",
            'Estimate_pp': est,
            'SE': se,
            'CI_low':  ci_low,
            'CI_high': ci_high,
            'z': z,
            'p_raw': p_raw,
        })

    out = pd.DataFrame(rows)
    if not out.empty:
        out['p_Holm'] = multipletests(out['p_raw'].values, method='holm')[1]
        out['Significant_Holm'] = out['p_Holm'] < 0.05
    return out


def filter_target_centric(df_contrasts, target='Adaptive'):
    """Keep only contrasts that include `target` on either side, sorted by p_Holm."""
    mask = df_contrasts['Contrast'].str.contains(target)
    out = df_contrasts.loc[mask].copy()
    if 'p_Holm' in out.columns:
        out = out.sort_values('p_Holm').reset_index(drop=True)
    return out
