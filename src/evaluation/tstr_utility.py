"""
TSTR (Train-on-Synthetic, Test-on-Real) utility evaluator.

Implements the protocol in Manuscript v2.3 Section III-D:
    - Four classifiers: XGBoost, LightGBM, RandomForest, LogisticRegression
    - Two targets: is_mi (positive rate ~3.2%), is_stroke (~5.1%)
    - Five seeds × 10-fold stratified cross-validation
    - Two subsets:
        * All     — full held-out fold
        * Tail P90 — patients with any HK variable ≥ 90th percentile
    - Six metrics: AUROC, AUPRC, Precision, Recall, F1, Brier
    - 95% bootstrap confidence intervals (N_BOOTSTRAP iterations, default 1000)

Real Baseline (TRTR): train on real fold, test on the held-out fold.

Each classifier uses its library's default hyperparameters with one
exception: standard imbalance handling is enabled (scale_pos_weight for
XGBoost, is_unbalance for LightGBM, class_weight='balanced' for Random
Forest and Logistic Regression) because the positive prevalence is below
6%. The same classifier configuration is used across all scalers and the
Real Baseline; no per-scaler tuning is performed.
"""

import os
import glob
import re
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_score, recall_score,
    f1_score, brier_score_loss, confusion_matrix,
)
from sklearn.utils import resample

# Optional gradient-boosting libraries; raise informative error if missing.
try:
    from xgboost import XGBClassifier
except ImportError as e:
    raise ImportError("xgboost is required for TSTR evaluation.") from e
try:
    from lightgbm import LGBMClassifier
except ImportError as e:
    raise ImportError("lightgbm is required for TSTR evaluation.") from e

import warnings
warnings.filterwarnings('ignore')


class TSTRUtilityEvaluator:
    """
    Compute TSTR / TRTR utility across scalers, seeds, folds, classifiers,
    targets, and subsets. Saves both raw per-fold results and bootstrap CIs.
    """

    HK_FEATURES = ['acr_ur', 'malb_ur', 'crtn_s', 'ggt_s', 'ins_s', 'tbil_s', 'ast_s']

    def __init__(self, cfg, fold_dir, fidelity_dir, output_dir):
        """
        Parameters
        ----------
        cfg          : parsed config dict
        fold_dir     : directory containing real_f{1..K}.csv
        fidelity_dir : directory containing per-scaler generated_data/ folders
        output_dir   : directory for TSTR CSV outputs
        """
        self.cfg = cfg
        self.fold_dir     = fold_dir
        self.fidelity_dir = fidelity_dir
        self.output_dir   = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        self.n_folds  = int(cfg.get('KFOLD_N', 10))
        self.targets  = cfg.get('TARGET_LABELS', ['is_mi', 'is_stroke'])
        self.cond     = cfg.get('COND_VARS', [])
        self.seeds    = cfg.get('SEEDS', [42, 52, 62, 72, 82])
        self.n_boot   = int(cfg.get('N_BOOTSTRAP', 1000))

        sample = pd.read_csv(os.path.join(self.fold_dir, 'real_f1.csv'))
        self.all_columns = list(sample.columns)
        exclude = set(self.cond + cfg.get('DROP_COLS', []) + ['p_id_group'])
        self.feature_cols = [
            c for c in self.all_columns
            if c not in exclude
            and sample[c].dtype in ['int64', 'float64', 'int32', 'float32', 'bool']
        ]

    # ------------------------------------------------------------------ #
    # Classifier factory                                                 #
    # ------------------------------------------------------------------ #
    # All classifiers use their library defaults except for an imbalance
    # flag (scale_pos_weight / is_unbalance / class_weight='balanced').
    # No further tuning is performed; the same configuration applies to
    # every scaler condition and to the Real Baseline.
    # ------------------------------------------------------------------ #
    @staticmethod
    def _build_classifiers(y_train):
        pos = int(np.sum(y_train))
        neg = len(y_train) - pos
        spw = (neg / pos) if pos > 0 else 1.0
        return {
            'XGBoost': XGBClassifier(
                scale_pos_weight=spw,
                random_state=42, eval_metric='logloss', verbosity=0,
            ),
            'LightGBM': LGBMClassifier(
                is_unbalance=True,
                random_state=42, verbosity=-1,
            ),
            'RandomForest': RandomForestClassifier(
                class_weight='balanced', random_state=42,
            ),
            'LogisticRegression': LogisticRegression(
                class_weight='balanced', random_state=42,
                max_iter=1000,   # default 100 fails to converge on clinical data
            ),
        }

    # ------------------------------------------------------------------ #
    # Metric computation                                                 #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _metrics(y_true, y_prob):
        y_pred = (y_prob >= 0.5).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        denom_spec = tn + fp
        return {
            'AUROC':    float(roc_auc_score(y_true, y_prob)),
            'AUPRC':    float(average_precision_score(y_true, y_prob)),
            'Precision': float(precision_score(y_true, y_pred, zero_division=0)),
            'Recall':    float(recall_score(y_true, y_pred, zero_division=0)),
            'F1':        float(f1_score(y_true, y_pred, zero_division=0)),
            'Brier':     float(brier_score_loss(y_true, y_prob)),
            'Specificity': float(tn / denom_spec) if denom_spec > 0 else 0.0,
        }

    def _bootstrap_ci(self, df_subset, metric_cols):
        """Percentile bootstrap 95% CI over rows of df_subset."""
        rng = np.random.RandomState(42)
        n = len(df_subset)
        if n < 2:
            return {m: (np.nan, np.nan, np.nan) for m in metric_cols}
        boot = {m: [] for m in metric_cols}
        for _ in range(self.n_boot):
            idx = rng.randint(0, n, size=n)
            sample = df_subset.iloc[idx]
            for m in metric_cols:
                boot[m].append(sample[m].mean())
        result = {}
        for m, vals in boot.items():
            arr = np.asarray(vals)
            result[m] = (
                float(np.mean(arr)),
                float(np.percentile(arr, 2.5)),
                float(np.percentile(arr, 97.5)),
            )
        return result

    # ------------------------------------------------------------------ #
    # Tail P90 mask                                                      #
    # ------------------------------------------------------------------ #
    def _tail_p90_mask(self, df_test):
        hk_in_test = [v for v in self.HK_FEATURES if v in df_test.columns]
        if not hk_in_test:
            return np.zeros(len(df_test), dtype=bool)
        mask = np.zeros(len(df_test), dtype=bool)
        for v in hk_in_test:
            mask = mask | (df_test[v].values >= df_test[v].quantile(0.90))
        return mask

    # ------------------------------------------------------------------ #
    # Real Baseline (TRTR)                                               #
    # ------------------------------------------------------------------ #
    def evaluate_real_baseline(self):
        rows = []
        for test_fold in range(1, self.n_folds + 1):
            df_test = pd.read_csv(
                os.path.join(self.fold_dir, f'real_f{test_fold}.csv')
            )
            df_train = pd.concat([
                pd.read_csv(os.path.join(self.fold_dir, f'real_f{i}.csv'))
                for i in range(1, self.n_folds + 1) if i != test_fold
            ], ignore_index=True)

            for target in self.targets:
                feats = [c for c in self.feature_cols if c != target]
                X_tr, y_tr = df_train[feats], df_train[target]
                X_te, y_te = df_test[feats],  df_test[target]
                tail_mask  = self._tail_p90_mask(df_test)

                for clf_name, clf in self._build_classifiers(y_tr).items():
                    try:
                        clf.fit(X_tr, y_tr)
                        p_all = clf.predict_proba(X_te)[:, 1]
                        m = self._metrics(y_te.values, p_all)
                        m.update({
                            'Condition': 'RealBaseline_TRTR',
                            'Seed': 0, 'Fold': test_fold,
                            'Target': target, 'Classifier': clf_name,
                            'Subset': 'All',
                        })
                        rows.append(m)

                        if tail_mask.sum() >= 5 and y_te[tail_mask].nunique() >= 2:
                            p_t = clf.predict_proba(X_te[tail_mask])[:, 1]
                            m_t = self._metrics(y_te[tail_mask].values, p_t)
                            m_t.update({
                                'Condition': 'RealBaseline_TRTR',
                                'Seed': 0, 'Fold': test_fold,
                                'Target': target, 'Classifier': clf_name,
                                'Subset': 'Tail_P90',
                            })
                            rows.append(m_t)
                    except Exception as exc:
                        print(f"[RealBaseline] {clf_name} fold{test_fold} "
                              f"{target}: {exc}")
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------ #
    # TSTR for one scaler condition                                      #
    # ------------------------------------------------------------------ #
    def evaluate_scaler(self, scaler_tag, gen_data_pattern):
        """
        Parameters
        ----------
        scaler_tag        : label (e.g. 'standard_base', 'adaptive_alpha_0_3')
        gen_data_pattern  : glob pattern matching generated CSVs in fidelity_dir
                            (one CSV per seed × fold)
        """
        all_rows = []
        for seed in self.seeds:
            for test_fold in range(1, self.n_folds + 1):
                # Locate generated synthetic file for (seed, fold)
                syn_path = self._find_syn(gen_data_pattern, seed, test_fold)
                if syn_path is None:
                    print(f"[TSTR] missing syn: seed={seed} fold={test_fold} "
                          f"tag={scaler_tag}")
                    continue
                df_syn  = pd.read_csv(syn_path)
                df_test = pd.read_csv(
                    os.path.join(self.fold_dir, f'real_f{test_fold}.csv')
                )

                for target in self.targets:
                    if target not in df_syn.columns or target not in df_test.columns:
                        continue
                    feats = [c for c in self.feature_cols if c != target
                             and c in df_syn.columns]
                    X_tr, y_tr = df_syn[feats], df_syn[target]
                    if y_tr.nunique() < 2:
                        continue
                    X_te, y_te = df_test[feats], df_test[target]
                    tail_mask = self._tail_p90_mask(df_test)

                    for clf_name, clf in self._build_classifiers(y_tr).items():
                        try:
                            clf.fit(X_tr, y_tr)
                            p_all = clf.predict_proba(X_te)[:, 1]
                            m = self._metrics(y_te.values, p_all)
                            m.update({
                                'Condition': scaler_tag,
                                'Seed': seed, 'Fold': test_fold,
                                'Target': target, 'Classifier': clf_name,
                                'Subset': 'All',
                            })
                            all_rows.append(m)

                            if tail_mask.sum() >= 5 and y_te[tail_mask].nunique() >= 2:
                                p_t = clf.predict_proba(X_te[tail_mask])[:, 1]
                                m_t = self._metrics(y_te[tail_mask].values, p_t)
                                m_t.update({
                                    'Condition': scaler_tag,
                                    'Seed': seed, 'Fold': test_fold,
                                    'Target': target, 'Classifier': clf_name,
                                    'Subset': 'Tail_P90',
                                })
                                all_rows.append(m_t)
                        except Exception as exc:
                            print(f"[TSTR] {scaler_tag} seed{seed} fold{test_fold} "
                                  f"{target} {clf_name}: {exc}")
        return pd.DataFrame(all_rows)

    # ------------------------------------------------------------------ #
    # Synthetic file lookup                                              #
    # ------------------------------------------------------------------ #
    def _find_syn(self, pattern, seed, fold):
        """
        Search for a synthetic CSV inside `pattern` matching the (seed, fold).
        Convention: filename ends with `_<seed>_<w>_syn_fold<fold>.csv`.
        """
        candidates = glob.glob(os.path.join(
            self.fidelity_dir, pattern, '**', f'*_{seed}_*_syn_fold{fold}.csv'
        ), recursive=True)
        if not candidates:
            return None
        candidates.sort()
        return candidates[0]

    # ------------------------------------------------------------------ #
    # Aggregation                                                        #
    # ------------------------------------------------------------------ #
    def aggregate_with_ci(self, df_results, out_path):
        """
        Aggregate per-row results into mean + 95% bootstrap CI per
        (Condition, Target, Classifier, Subset). Saves CSV in the format
        used by the manuscript tables.
        """
        if df_results.empty:
            print("[aggregate] empty results, skipping.")
            return
        metric_cols = ['AUROC', 'AUPRC', 'Precision', 'Recall',
                       'F1', 'Brier', 'Specificity']
        keys = ['Condition', 'Target', 'Classifier', 'Subset']
        rows = []
        for (cond, tgt, clf, sub), grp in df_results.groupby(keys):
            ci = self._bootstrap_ci(grp, metric_cols)
            row = {'Condition': cond, 'Target': tgt,
                   'Classifier': clf, 'Subset': sub, 'N': len(grp)}
            for m, (mean, lo, hi) in ci.items():
                row[f'{m}_mean'] = mean
                row[f'{m}_ci_low']  = lo
                row[f'{m}_ci_high'] = hi
            rows.append(row)
        pd.DataFrame(rows).to_csv(out_path, index=False)
        print(f"[aggregate] saved: {out_path}")
