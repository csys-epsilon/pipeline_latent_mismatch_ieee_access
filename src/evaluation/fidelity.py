"""
Fidelity evaluation for synthetic clinical data.

Implements the pipeline-level fidelity metrics described in Manuscript v2.3,
Section III-D ("Evaluation Framework"):
    - Kolmogorov-Smirnov (KS) distance
    - KL divergence (histogram-based)
    - Standardized mean difference (SMD)
    - Wasserstein distance (1-D)
    - Skewness / kurtosis (real vs. synthetic)
    - Quantile preservation at {P50, P90, P95, P97.5, P99}
    - Kurtosis preservation rate (Section IV principal tail metric)

The kurtosis preservation rate is the ratio of synthetic-to-real kurtosis
expressed as a percentage. Values below 100 indicate under-reproduction,
above 100 over-reproduction.
"""

import os
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import (
    ks_2samp, entropy, wasserstein_distance, skew, kurtosis
)


class FidelityEvaluator:

    def __init__(self, config, num_features):
        self.config = config
        self.target_labels = config.get('TARGET_LABELS', [])
        self.num_features  = num_features

    # ------------------------------------------------------------------ #
    # Per-variable univariate metrics                                    #
    # ------------------------------------------------------------------ #
    def analysis_univariate(self, df_real, df_syn):
        """
        Returns
        -------
        metrics_df   : per-variable distributional metrics
        quantiles_df : per-variable quantile values (Real vs Syn)
        corr_real    : correlation matrix on real data
        corr_syn     : correlation matrix on synthetic data
        """
        metrics_list, quantiles_list = [], []
        equiv_margin = float(self.config.get('EQUIV_MARGIN', 0.2))

        for col in self.num_features:
            if col not in df_real.columns or col not in df_syn.columns:
                continue

            r = df_real[col].values
            s = df_syn[col].values

            mean_r, mean_s = float(np.mean(r)), float(np.mean(s))
            var_r,  var_s  = float(np.var(r)),  float(np.var(s))

            # Standardized mean difference
            smd = abs(mean_r - mean_s) / np.sqrt((var_r + var_s) / 2.0 + 1e-10)

            # Welch's t-test
            try:
                t_p = float(stats.ttest_ind(r, s, equal_var=False).pvalue)
            except Exception:
                t_p = 1.0

            # TOST equivalence test
            try:
                diff = mean_r - mean_s
                se = np.sqrt(var_r / len(r) + var_s / len(s) + 1e-10)
                df_t = len(r) + len(s) - 2
                p1 = 1.0 - stats.t.cdf((diff + equiv_margin) / se, df=df_t)
                p2 = stats.t.cdf((diff - equiv_margin) / se, df=df_t)
                tost_p = float(max(p1, p2))
            except Exception:
                tost_p = 1.0

            # Kolmogorov-Smirnov distance
            try:
                ks_stat = float(ks_2samp(r, s).statistic)
            except Exception:
                ks_stat = 0.0

            # Wasserstein distance (1-D)
            try:
                w_dist = float(wasserstein_distance(r, s))
            except Exception:
                w_dist = 0.0

            # KL divergence (histogram-based, shared bins from real data)
            try:
                hist_r, bins = np.histogram(r, bins=50, density=True)
                hist_s, _ = np.histogram(s, bins=bins, density=True)
                kl_div = float(entropy(hist_r + 1e-10, hist_s + 1e-10))
            except Exception:
                kl_div = 0.0

            # Skewness and (excess) kurtosis — Fisher's definition (matches manuscript)
            try:
                skew_r, skew_s = float(skew(r)), float(skew(s))
                kurt_r, kurt_s = float(kurtosis(r, fisher=True)), float(kurtosis(s, fisher=True))
            except Exception:
                skew_r = skew_s = kurt_r = kurt_s = 0.0

            metrics_list.append({
                'Feature': col,
                'SMD': smd,
                't_test_p': t_p,
                'TOST_p': tost_p,
                'KS': ks_stat,
                'Wasserstein': w_dist,
                'KL_Divergence': kl_div,
                'Skew_Real': skew_r, 'Skew_Syn': skew_s,
                'Kurtosis_Real': kurt_r, 'Kurtosis_Syn': kurt_s,
            })

            # Quantile preservation at standard tail percentiles
            quantile_levels = {0.5: 'P50', 0.9: 'P90', 0.95: 'P95',
                               0.975: 'P97.5', 0.99: 'P99'}
            for q, name in quantile_levels.items():
                try:
                    rq = float(np.percentile(r, q * 100)) if len(r) >= 50 else np.nan
                    sq = float(np.percentile(s, q * 100)) if len(s) >= 50 else np.nan
                except Exception:
                    rq = sq = np.nan
                quantiles_list.append({
                    'Feature': col, 'Segment': name, 'Real': rq, 'Syn': sq
                })

        try:
            corr_r = df_real[self.num_features].corr()
            corr_s = df_syn[self.num_features].corr()
        except Exception:
            corr_r = corr_s = None

        return (pd.DataFrame(metrics_list), pd.DataFrame(quantiles_list),
                corr_r, corr_s)

    # ------------------------------------------------------------------ #
    # Report saving                                                      #
    # ------------------------------------------------------------------ #
    def save_report(self, all_metrics, all_quantiles, out_dir, scaler_name, tag=''):
        """
        Concatenate fold-level metrics and write three CSVs:
            report_metrics_<scaler>{tag}.csv               (per-fold raw metrics)
            report_quantiles_<scaler>{tag}.csv             (per-fold quantiles)
            report_kurtosis_preservation_<scaler>{tag}.csv (manuscript metric)

        Kurtosis preservation rate follows Manuscript v2.3 Section III-D:
            "Kurtosis values are computed as fold-mean aggregated kurtosis."
        That is, per Feature:
            preservation_rate = mean_over_folds(Kurt_Syn) / mean_over_folds(Kurt_Real) * 100
        This is a ratio-of-means (RoM), NOT a mean-of-ratios (MoR). The two
        differ when the real kurtosis varies across folds; the RoM matches
        the manuscript's reported HK-mean values (e.g. 135.1% for Quantile,
        82.4% for Power, 35.6% for Adaptive at α=0.3 / 7-HK).
        """
        os.makedirs(out_dir, exist_ok=True)
        suffix = f"_{tag}" if tag else ""

        df_m = pd.concat(all_metrics, ignore_index=True) if all_metrics else pd.DataFrame()
        df_q = pd.concat(all_quantiles, ignore_index=True) if all_quantiles else pd.DataFrame()

        if not df_m.empty:
            # Per-fold raw metrics (audit trail)
            df_m.to_csv(
                os.path.join(out_dir, f'report_metrics_{scaler_name}{suffix}.csv'),
                index=False,
            )

            # Fold-mean aggregated kurtosis (manuscript-style)
            agg = (
                df_m.groupby('Feature', as_index=False)
                    .agg(Kurt_Real_mean=('Kurtosis_Real', 'mean'),
                         Kurt_Syn_mean =('Kurtosis_Syn',  'mean'))
            )
            denom = agg['Kurt_Real_mean'].replace(0, np.nan)
            agg['Kurtosis_Preservation_Rate'] = (
                agg['Kurt_Syn_mean'] / denom
            ) * 100.0
            agg = agg.sort_values('Feature').reset_index(drop=True)

            agg.to_csv(
                os.path.join(out_dir, f'report_kurtosis_preservation_{scaler_name}{suffix}.csv'),
                index=False,
            )

        if not df_q.empty:
            df_q.to_csv(
                os.path.join(out_dir, f'report_quantiles_{scaler_name}{suffix}.csv'),
                index=False,
            )
