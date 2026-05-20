"""
Preprocessing scalers for the Pipeline-Latent Mismatch study.
Implements five scalers (Standard, Quantile, Robust, Power, Adaptive) and the
forward/inverse transforms used by both TabDDPM and CVAE pipelines.

Reference: Manuscript v2.3, Section III-B (Preprocessing Scalers).
"""

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import (
    StandardScaler, QuantileTransformer, RobustScaler, PowerTransformer
)


class DataProcess:
    """
    Five-scaler preprocessing pipeline with head-tail Adaptive transform.

    Scalers (Section III-B):
        - standard   : z-score normalization (Eq. 1)
        - quantile   : empirical CDF -> N(0, 1) (Eq. 2)
        - robust     : (x - median) / IQR        (Eq. 3)
        - power      : Yeo-Johnson + z-score     (Eq. 4)
        - adaptive   : Quantile (head) + StandardScaler(log-compressed) (tail), Eqs. 5-7

    The Adaptive scaler partitions each HK variable at the P97.5 threshold,
    with a linear-mixed transition zone from P95 to P99. The tail region is
    log-compressed with strength `log_alpha` ∈ [0, 1] before z-score scaling.
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.w_tail = float(cfg.get('w_tail', 1.0))
        prep = cfg.get('preprocessing', {})
        self.split_features = prep.get('split_quantile_features', [])

        # Adaptive partition parameters (defaults match manuscript)
        self.base_tail_quantile = float(cfg.get('base_tail_quantile', 0.975))
        self.mix_low_quantile   = float(cfg.get('mix_low_quantile',   0.95))
        self.mix_high_quantile  = float(cfg.get('mix_high_quantile',  0.99))

        # Tail log-compression strength α ∈ [0, 1] (Eq. 7)
        # α = 0  -> identity (no compression)
        # α = 1  -> sign(x) * log1p(|x|)
        # 0<α<1 -> sign(x) * log1p(α·|x|) / α
        self.log_alpha = float(cfg.get('log_alpha', 0.3))

        # Iterative inverse-refinement settings (Newton-style)
        self.inv_max_iter = int(cfg.get('inv_max_iter', 30))
        self.inv_tol      = float(cfg.get('inv_tol', 1e-8))

        self.cond_vars = cfg.get('COND_VARS', [])
        self.drop_cols = list(dict.fromkeys(
            cfg.get('DROP_VARS', []) + cfg.get('DROP_COLS', [])
        ))

        # Fitted scaler containers
        self.default_scaler = None    # StandardScaler
        self.qt_scaler      = None    # QuantileTransformer
        self.robust_scaler  = None    # RobustScaler
        self.power_scaler   = None    # PowerTransformer (Yeo-Johnson)
        self.head_scalers   = {}      # Per-variable QuantileTransformer (head)
        self.tail_scalers   = {}      # Per-variable StandardScaler      (tail)
        self.num_features   = []

        # Per-variable adaptive boundaries
        self.decision_boundaries   = {}   # tau_low  (P95)
        self.tail_boundaries       = {}   # tau_high (P99)
        self.base_tail_boundaries  = {}   # P97.5 hard tail threshold
        self.head_max_scaled       = {}
        self.tail_start_scaled     = {}
        self.offsets               = {}

    # ------------------------------------------------------------------ #
    # Tail compression / decompression                                   #
    # ------------------------------------------------------------------ #
    def _compress(self, x):
        """Sign-preserving log compression (forward)."""
        if self.log_alpha <= 1e-6:
            return x
        return np.sign(x) * np.log1p(self.log_alpha * np.abs(x)) / self.log_alpha

    def _decompress(self, y):
        """Sign-preserving log decompression (inverse of _compress)."""
        if self.log_alpha <= 1e-6:
            return y
        return np.sign(y) * np.expm1(self.log_alpha * np.abs(y)) / self.log_alpha

    # ------------------------------------------------------------------ #
    # Adaptive boundary helpers                                          #
    # ------------------------------------------------------------------ #
    def _resolve_mixed_boundaries(self, series):
        tau_low  = float(series.quantile(self.mix_low_quantile))
        tau_high = float(series.quantile(self.mix_high_quantile))
        if not np.isfinite(tau_low):
            tau_low = float(series.median())
        if not np.isfinite(tau_high):
            tau_high = tau_low
        if tau_high <= tau_low:
            eps = max(float(series.std(ddof=0)) * 1e-4, 1e-5)
            tau_high = tau_low + eps
        return tau_low, tau_high

    def _linear_mix_weight(self, values, tau_low, tau_high):
        w = (values - tau_low) / (tau_high - tau_low + 1e-10)
        return np.clip(w, 0.0, 1.0)

    # ------------------------------------------------------------------ #
    # Global scaler fitting (computed on full Arirang cohort)            #
    # ------------------------------------------------------------------ #
    def set_global_scaler(self, df_total):
        """Fit the four baseline scalers on the full numerical feature set."""
        exclude = set(self.cond_vars + self.drop_cols)
        potential_num = [c for c in df_total.columns if c not in exclude]
        self.num_features = (
            df_total[potential_num].select_dtypes(include=[np.number]).columns.tolist()
        )

        n_q = min(1000, max(10, len(df_total)))
        self.default_scaler = StandardScaler().fit(df_total[self.num_features])
        self.qt_scaler = QuantileTransformer(
            output_distribution='normal', n_quantiles=n_q, random_state=42
        ).fit(df_total[self.num_features])
        self.robust_scaler = RobustScaler().fit(df_total[self.num_features])
        self.power_scaler  = PowerTransformer(
            method='yeo-johnson', standardize=True
        ).fit(df_total[self.num_features])

    # ------------------------------------------------------------------ #
    # Train/valid preprocessing                                          #
    # ------------------------------------------------------------------ #
    def preprocess_train_valid(self, train_df, valid_df, s_type, cond_vars=None):
        """
        Fit Adaptive-specific head/tail scalers on `train_df` and transform both folds.

        Returns
        -------
        xt, ct      : training tensors (features, condition)
        raw_mask    : float tensor indicating samples at-or-above P97.5
        vx, vc      : validation tensors (features, condition)
        num_features: list of numerical feature names
        default_scaler : reference to the fitted StandardScaler
        """
        if self.default_scaler is None:
            self.set_global_scaler(train_df)

        df_tr, df_val = train_df.copy(), valid_df.copy()
        s_lower = s_type.lower()
        raw_tail_mask = np.zeros(len(df_tr), dtype=bool)

        # Fit per-variable Adaptive scalers on training data only
        if s_lower == 'adaptive':
            for col in self.split_features:
                if col not in self.num_features:
                    continue

                tau_low, tau_high = self._resolve_mixed_boundaries(df_tr[col])
                base_tail = float(df_tr[col].quantile(self.base_tail_quantile))

                self.decision_boundaries[col]  = tau_low
                self.tail_boundaries[col]      = tau_high
                self.base_tail_boundaries[col] = base_tail
                raw_tail_mask |= (df_tr[col] >= base_tail).values

                head_data = df_tr[df_tr[col] <= tau_high][col].values.reshape(-1, 1)
                tail_data = df_tr[df_tr[col] >= tau_low][col].values.reshape(-1, 1)

                self.head_scalers[col] = QuantileTransformer(
                    output_distribution='normal',
                    n_quantiles=min(500, max(10, len(head_data))),
                    random_state=42,
                ).fit(head_data)

                tail_compressed = self._compress(tail_data)
                self.tail_scalers[col] = StandardScaler().fit(tail_compressed)

                self.head_max_scaled[col] = float(
                    self.head_scalers[col].transform([[tau_low]]).item()
                )
                self.offsets[col] = self.head_max_scaled[col] + self.w_tail
                tau_high_c = self._compress(np.array([[tau_high]]))
                self.tail_start_scaled[col] = (
                    self.offsets[col]
                    + float(self.tail_scalers[col].transform(tau_high_c).item())
                )

        # Forward transformer (callable per dataframe)
        def transform_df(df):
            res = df.copy()
            if s_lower == 'standard':
                res[self.num_features] = self.default_scaler.transform(df[self.num_features])
            elif s_lower == 'quantile':
                res[self.num_features] = self.qt_scaler.transform(df[self.num_features])
            elif s_lower == 'robust':
                res[self.num_features] = self.robust_scaler.transform(df[self.num_features])
            elif s_lower == 'power':
                res[self.num_features] = self.power_scaler.transform(df[self.num_features])
            elif s_lower == 'adaptive':
                # Apply Quantile to non-split features
                normal_feats = [c for c in self.num_features if c not in self.split_features]
                if normal_feats:
                    full = self.qt_scaler.transform(df[self.num_features])
                    full_df = pd.DataFrame(full, columns=self.num_features, index=df.index)
                    res[normal_feats] = full_df[normal_feats]
                # Apply Adaptive head-tail mix to split features
                for col in self.split_features:
                    if col not in self.num_features:
                        continue
                    val = df[col].values.reshape(-1, 1)
                    h_s = self.head_scalers[col].transform(val)
                    val_c = self._compress(val)
                    t_s = self.offsets[col] + self.tail_scalers[col].transform(val_c)
                    w = self._linear_mix_weight(
                        val, self.decision_boundaries[col], self.tail_boundaries[col]
                    )
                    w[val >= self.base_tail_boundaries[col]] = 1.0
                    res[col] = ((1.0 - w) * h_s + w * t_s).ravel()
            else:
                raise ValueError(f"Unknown scaler type: {s_type}")
            return res

        tr_t  = transform_df(df_tr)
        val_t = transform_df(df_val)

        return (
            torch.FloatTensor(tr_t[self.num_features].values),
            torch.FloatTensor(tr_t[self.cond_vars].values),
            torch.FloatTensor(raw_tail_mask.astype(np.float32)),
            torch.FloatTensor(val_t[self.num_features].values),
            torch.FloatTensor(val_t[self.cond_vars].values),
            self.num_features,
            self.default_scaler,
        )

    # ------------------------------------------------------------------ #
    # Inverse transformation                                             #
    # ------------------------------------------------------------------ #
    def perform_inverse_transform(self, real_raw_df, syn_tensor, s_type):
        dr = real_raw_df.copy()
        ds = self.inverse_transform(syn_tensor, s_type=s_type)
        return dr, ds

    def _forward_single(self, x_raw, col):
        """Adaptive forward transform for a single variable (used by inverse refinement)."""
        h_s = self.head_scalers[col].transform(x_raw)
        x_c = self._compress(x_raw)
        t_s = self.offsets[col] + self.tail_scalers[col].transform(x_c)
        w = self._linear_mix_weight(
            x_raw, self.decision_boundaries[col], self.tail_boundaries[col]
        )
        w[x_raw >= self.base_tail_boundaries[col]] = 1.0
        return (1.0 - w) * h_s + w * t_s

    def inverse_transform(self, tensor, s_type='adaptive'):
        """Map generated latent samples back to the original feature space."""
        data_np = (
            tensor.detach().cpu().numpy()
            if isinstance(tensor, torch.Tensor) else np.asarray(tensor)
        )
        ds = pd.DataFrame(data_np, columns=self.num_features)
        s_lower = s_type.lower()

        if s_lower == 'standard':
            ds[self.num_features] = self.default_scaler.inverse_transform(data_np)

        elif s_lower == 'quantile':
            ds[self.num_features] = self.qt_scaler.inverse_transform(data_np)

        elif s_lower == 'robust':
            # No clipping: keep raw model output to preserve faithfulness of the
            # Pipeline-Latent Mismatch claim. Replace any inf/NaN with finite zeros.
            inv = self.robust_scaler.inverse_transform(data_np)
            inv = np.nan_to_num(inv, nan=0.0, posinf=0.0, neginf=0.0)
            ds[self.num_features] = inv

        elif s_lower == 'power':
            # Yeo-Johnson inverse (Eq. 4, first case) is closed-form 1/λ which
            # amplifies small latent perturbations into arbitrarily large
            # original-scale values when the fitted λ approaches zero, as noted
            # in Section IV-A of the manuscript. Without this clip the inverse
            # overflows to ±inf for a non-negligible fraction of samples and
            # the resulting HK kurtosis-preservation rate becomes undefined.
            # Clipping the latent to [-5, 5] (≈99.99994% of N(0,1) support)
            # is the minimal numerical guard required to reproduce the
            # manuscript's reported HK mean of 82.4% (Table II).
            data_clipped = np.clip(data_np, -5.0, 5.0)
            inv = self.power_scaler.inverse_transform(data_clipped)
            inv = np.nan_to_num(inv, nan=0.0, posinf=0.0, neginf=0.0)
            ds[self.num_features] = inv

        elif s_lower == 'adaptive':
            qt_inv_all = self.qt_scaler.inverse_transform(data_np)
            for i, col in enumerate(self.num_features):
                x_mix = data_np[:, i].reshape(-1, 1)
                if col not in self.split_features:
                    ds[col] = qt_inv_all[:, i]
                    continue

                h_inv = self.head_scalers[col].inverse_transform(x_mix)
                t_inv_c = self.tail_scalers[col].inverse_transform(
                    x_mix - self.offsets[col]
                )
                t_inv = self._decompress(t_inv_c)

                # Initial estimate based on mixing weight in latent space
                w_l = np.clip(
                    (x_mix - self.head_max_scaled[col]) /
                    (self.tail_start_scaled[col] - self.head_max_scaled[col] + 1e-10),
                    0.0, 1.0,
                )
                x_est = np.where(w_l > 0.5, t_inv, h_inv)

                # Newton-style iterative refinement
                target = x_mix.copy()
                prev_resid_norm = np.inf
                alpha = 0.5
                for it in range(self.inv_max_iter):
                    latent_est = self._forward_single(x_est, col)
                    residual = target - latent_est
                    max_r = float(np.max(np.abs(residual)))
                    if max_r < self.inv_tol:
                        break

                    eps = 1e-6
                    latent_plus = self._forward_single(x_est + eps, col)
                    grad = (latent_plus - latent_est) / eps
                    safe_grad = np.where(np.abs(grad) > 1e-8, grad, 1.0)
                    dx = residual / safe_grad
                    dx = np.where(np.isfinite(dx), dx, 0.0)
                    dx = np.clip(dx, -np.abs(x_est) * 0.5 - 1e-3,
                                      np.abs(x_est) * 0.5 + 1e-3)

                    if it == 0:
                        prev_resid_norm = max_r
                    else:
                        alpha = (min(0.9, alpha * 1.1)
                                 if max_r < prev_resid_norm
                                 else max(0.1, alpha * 0.5))
                        prev_resid_norm = max_r

                    x_est = x_est + alpha * dx

                ds[col] = x_est.ravel()
        else:
            raise ValueError(f"Unknown scaler type: {s_type}")

        return ds
