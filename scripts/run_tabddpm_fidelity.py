"""
TabDDPM fidelity evaluation pipeline.

Trains TabDDPM on each (scaler × seed × fold) configuration, generates
synthetic data of size = held-out fold size using train-fold conditional
vectors, runs the inverse transform, and saves per-fold synthetic CSVs
and univariate fidelity metrics.

Usage
-----
    python scripts/run_tabddpm_fidelity.py --config configs/config.yaml

To run the Adaptive α sweep, edit configs/config.yaml:
    SCALERS: ['adaptive']
    ENABLE_ALPHA_SWEEP: true
"""

import argparse
import os
import random
import sys
import yaml

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

# Make `src/` importable when running from repo root.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.preprocessing.scalers import DataProcess
from src.models.tabddpm import TabDDPMTrainer
from src.evaluation.fidelity import FidelityEvaluator


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_paths(cfg):
    """Build absolute paths from the config's PATHS section."""
    root = cfg['ROOT_DIR']
    paths = cfg.get('PATHS', {})
    exp = paths.get('experiment_condition', 'Arirang_All_vars')
    fid_model = paths.get('fidelity_model', 'tabDDPM')

    def _p(*parts):
        return os.path.join(root, *[p.replace('/', os.sep) for p in parts])

    return {
        'real_folds_dir':  _p(paths.get('real_folds_dir', 'raw_data/real_folds')),
        'fidelity_dir':    _p(paths.get('fidelity_root', 'evaluation/fidelity'),
                              fid_model, exp),
        'subdir_uni':      paths.get('subdir_univariate', 'eval_univariate'),
        'subdir_gen':      paths.get('subdir_generated',  'generated_data'),
        'experiment':      exp,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True, help='Path to YAML config')
    args = ap.parse_args()

    with open(args.config, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    paths = resolve_paths(cfg)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    n_folds = int(cfg.get('KFOLD_N', 10))
    seeds = cfg.get('SEEDS', [42, 52, 62, 72, 82])
    scalers = cfg.get('SCALERS', ['standard', 'quantile', 'robust', 'power', 'adaptive'])
    tabddpm_cfg = cfg.get('TABDDPM', {})
    enable_sweep = bool(cfg.get('ENABLE_ALPHA_SWEEP', False))
    alpha_sweep  = cfg.get('ALPHA_SWEEP', [0.0, 0.1, 0.3, 0.5, 0.7, 1.0])

    fold_dir = paths['real_folds_dir']
    print(f"[INFO] device={device} | folds={n_folds} | seeds={seeds}")
    print(f"[INFO] fidelity_dir = {paths['fidelity_dir']}")

    for scaler in scalers:
        # Adaptive alpha sweep; for non-Adaptive scalers use default log_alpha
        if enable_sweep and scaler == 'adaptive':
            alpha_values = alpha_sweep
        else:
            alpha_values = [cfg.get('log_alpha', 0.3)]

        for alpha_val in alpha_values:
            cfg['log_alpha'] = alpha_val

            if enable_sweep and scaler == 'adaptive':
                alpha_str = f"{alpha_val:.1f}".replace('.', '_')
                scaler_dir = os.path.join(
                    paths['fidelity_dir'], f'{scaler}_alpha', alpha_str
                )
            else:
                scaler_dir = os.path.join(paths['fidelity_dir'], f'{scaler}_base')
            os.makedirs(scaler_dir, exist_ok=True)

            for seed in seeds:
                set_seed(seed)
                exp_dir = os.path.join(scaler_dir, f'TabDDPM_{scaler}_Seed{seed}')
                gen_dir = os.path.join(exp_dir, paths['subdir_gen'])
                uni_dir = os.path.join(exp_dir, paths['subdir_uni'])
                os.makedirs(gen_dir, exist_ok=True)
                os.makedirs(uni_dir, exist_ok=True)

                # Fit global scalers. Two modes:
                #   - 'all_folds' (default, matches the manuscript's reported runs):
                #       Baseline scalers are fitted on the concatenation of every
                #       fold. This mirrors how the original experiments were run,
                #       and is acceptable because the same pre-fit is shared by
                #       every scaler condition (so the relative comparisons are
                #       not biased). Cf. README §"Configuration notes".
                #   - 'train_only' (strict, leakage-free):
                #       Baseline scalers are re-fit inside each fold loop using
                #       only the training folds. Use this for an externally
                #       auditable, leakage-free run.
                dp = DataProcess(cfg)
                global_fit_mode = cfg.get('GLOBAL_SCALER_FIT', 'all_folds')
                if global_fit_mode == 'all_folds':
                    pool = pd.concat([
                        pd.read_csv(os.path.join(fold_dir, f'real_f{i}.csv'))
                        for i in range(1, n_folds + 1)
                    ], ignore_index=True)
                    dp.set_global_scaler(pool)
                elif global_fit_mode != 'train_only':
                    raise ValueError(
                        f"Unknown GLOBAL_SCALER_FIT: {global_fit_mode!r}. "
                        "Choose 'all_folds' or 'train_only'."
                    )
                fid_eval = FidelityEvaluator(cfg, dp.num_features) \
                    if global_fit_mode == 'all_folds' else None

                fold_metrics, fold_quantiles = [], []
                for fold in range(1, n_folds + 1):
                    val_df = pd.read_csv(os.path.join(fold_dir, f'real_f{fold}.csv'))
                    train_df = pd.concat([
                        pd.read_csv(os.path.join(fold_dir, f'real_f{i}.csv'))
                        for i in range(1, n_folds + 1) if i != fold
                    ], ignore_index=True)

                    # In 'train_only' mode, re-fit the global baseline scalers
                    # on the training folds only (no test contamination).
                    if global_fit_mode == 'train_only':
                        dp = DataProcess(cfg)
                        dp.set_global_scaler(train_df)
                        fid_eval = FidelityEvaluator(cfg, dp.num_features)

                    # Preprocess (fits adaptive scalers on train only either way)
                    xt, ct, msk, vx, vc, num_f, _ = dp.preprocess_train_valid(
                        train_df, val_df, scaler
                    )

                    # Train TabDDPM (P97.5 tail mask `msk` is consumed by
                    # the trainer to apply the 2.5× heavy-tail weight).
                    trainer = TabDDPMTrainer(
                        len(num_f), len(cfg['COND_VARS']), tabddpm_cfg, device
                    )
                    bs = int(tabddpm_cfg.get('batch_size', 512))
                    loader = DataLoader(
                        TensorDataset(xt, ct, msk),
                        batch_size=bs, shuffle=True, drop_last=False,
                    )
                    n_epochs = int(tabddpm_cfg.get('epochs', 2000))
                    for ep in range(n_epochs):
                        for bx, bc, bm in loader:
                            trainer.train_step(
                                bx.to(device), bc.to(device), bm.to(device),
                            )
                        if (ep + 1) % 200 == 0:
                            print(f"  [{scaler}|s{seed}|f{fold}] epoch {ep+1}/{n_epochs}")

                    # Sample with train-fold conditional vectors, size = val fold size
                    n_syn = vc.size(0)
                    idx = torch.randperm(ct.size(0))[:n_syn]
                    ct_sampled = ct[idx].to(device)
                    gx = trainer.sample(ct_sampled)

                    # Inverse transform
                    dr, ds = dp.perform_inverse_transform(val_df, gx, scaler)
                    cond_df = pd.DataFrame(
                        ct_sampled.cpu().numpy(), columns=cfg['COND_VARS']
                    )
                    ds = pd.concat(
                        [ds.reset_index(drop=True), cond_df.reset_index(drop=True)],
                        axis=1,
                    )
                    keep = dp.num_features + list(cfg['COND_VARS'])
                    dr_keep = dr[keep].reset_index(drop=True)
                    ds_keep = ds[keep].reset_index(drop=True)

                    # Save synthetic CSV (downstream TSTR consumes these)
                    syn_path = os.path.join(
                        gen_dir, f'{scaler}_{seed}_1_syn_fold{fold}.csv'
                    )
                    ds_keep.to_csv(syn_path, index=False)

                    # Univariate fidelity
                    fm, fq, _, _ = fid_eval.analysis_univariate(dr_keep, ds_keep)
                    fm['seed'] = seed; fm['fold'] = fold
                    fq['seed'] = seed; fq['fold'] = fold
                    fold_metrics.append(fm)
                    fold_quantiles.append(fq)

                # Save per-seed aggregated metrics
                fid_eval.save_report(fold_metrics, fold_quantiles, exp_dir, scaler)
                print(f"[DONE] {scaler} | seed={seed} | dir={exp_dir}")


if __name__ == '__main__':
    main()
