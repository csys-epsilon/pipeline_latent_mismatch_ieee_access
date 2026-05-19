"""
CVAE fidelity evaluation pipeline.

Same train/sample/evaluate skeleton as run_tabddpm_fidelity.py, but using
the Conditional VAE generative model and the primary CVAE configuration
specified in Manuscript v2.3 Section III-C:
    hidden_dim 256, latent_dim 64, batch_size 512, 2000 epochs,
    Huber (δ=1.5) + β·KL with β = 0.01.

Capacity ablation: set CVAE.hidden_dim = 128 in config.
KL reweighting sweep: set CVAE.beta to one of {0.05, 0.1, 0.5, 1.0}
                      and CVAE.latent_dim = 32.

Usage
-----
    python scripts/run_cvae_fidelity.py --config configs/config.yaml
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

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.preprocessing.scalers import DataProcess
from src.models.cvae import CVAETrainer
from src.evaluation.fidelity import FidelityEvaluator


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_paths(cfg):
    root = cfg['ROOT_DIR']
    paths = cfg.get('PATHS', {})
    exp = paths.get('experiment_condition', 'Arirang_All_vars')
    # CVAE writes under fidelity/cvae/<exp> by default
    fid_model = paths.get('fidelity_model_cvae',
                          paths.get('fidelity_model', 'cvae'))

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

    # Force the CVAE pipeline to write under fidelity/cvae/<exp>
    cfg.setdefault('PATHS', {})['fidelity_model_cvae'] = 'cvae'

    paths = resolve_paths(cfg)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    n_folds = int(cfg.get('KFOLD_N', 10))
    seeds = cfg.get('SEEDS', [42, 52, 62, 72, 82])
    scalers = cfg.get('SCALERS', ['standard', 'quantile', 'robust', 'power', 'adaptive'])
    cvae_cfg = cfg.get('CVAE', {})

    fold_dir = paths['real_folds_dir']
    print(f"[INFO] device={device} | folds={n_folds} | seeds={seeds}")
    print(f"[INFO] CVAE hidden_dim={cvae_cfg.get('hidden_dim')}, "
          f"latent_dim={cvae_cfg.get('latent_dim')}, "
          f"batch={cvae_cfg.get('batch_size')}, beta={cvae_cfg.get('beta')}")
    print(f"[INFO] fidelity_dir = {paths['fidelity_dir']}")

    for scaler in scalers:
        scaler_dir = os.path.join(paths['fidelity_dir'], f'CVAE_{scaler}_base')
        os.makedirs(scaler_dir, exist_ok=True)

        for seed in seeds:
            set_seed(seed)
            exp_dir = os.path.join(scaler_dir, f'CVAE_{scaler}_Seed{seed}')
            gen_dir = os.path.join(exp_dir, paths['subdir_gen'])
            uni_dir = os.path.join(exp_dir, paths['subdir_uni'])
            os.makedirs(gen_dir, exist_ok=True)
            os.makedirs(uni_dir, exist_ok=True)

            # Fit global scalers. See run_tabddpm_fidelity.py for the rationale
            # behind the two modes. Defaults to 'all_folds' to match the
            # manuscript runs; switch to 'train_only' for a leakage-free run.
            dp = DataProcess(cfg)
            global_fit_mode = cfg.get('GLOBAL_SCALER_FIT', 'all_folds')
            if global_fit_mode == 'all_folds':
                pool = pd.concat([
                    pd.read_csv(os.path.join(fold_dir, f'real_f{i}.csv'))
                    for i in range(1, n_folds + 1)
                ], ignore_index=True)
                dp.set_global_scaler(pool)
                fid_eval = FidelityEvaluator(cfg, dp.num_features)
            elif global_fit_mode != 'train_only':
                raise ValueError(
                    f"Unknown GLOBAL_SCALER_FIT: {global_fit_mode!r}. "
                    "Choose 'all_folds' or 'train_only'."
                )

            fold_metrics, fold_quantiles = [], []
            for fold in range(1, n_folds + 1):
                val_df = pd.read_csv(os.path.join(fold_dir, f'real_f{fold}.csv'))
                train_df = pd.concat([
                    pd.read_csv(os.path.join(fold_dir, f'real_f{i}.csv'))
                    for i in range(1, n_folds + 1) if i != fold
                ], ignore_index=True)

                if global_fit_mode == 'train_only':
                    dp = DataProcess(cfg)
                    dp.set_global_scaler(train_df)
                    fid_eval = FidelityEvaluator(cfg, dp.num_features)

                xt, ct, msk, vx, vc, num_f, _ = dp.preprocess_train_valid(
                    train_df, val_df, scaler
                )

                trainer = CVAETrainer(
                    len(num_f), len(cfg['COND_VARS']), cvae_cfg, device
                )
                bs = int(cvae_cfg.get('batch_size', 512))
                loader = DataLoader(
                    TensorDataset(xt, ct, msk),
                    batch_size=bs, shuffle=True, drop_last=False,
                )
                n_epochs = int(cvae_cfg.get('epochs', 2000))
                for ep in range(n_epochs):
                    for bx, bc, bm in loader:
                        trainer.train_step(
                            bx.to(device), bc.to(device), bm.to(device),
                        )
                    if (ep + 1) % 200 == 0:
                        # Quick log: one mini-batch loss (with mask for parity)
                        with torch.no_grad():
                            bx, bc, bm = next(iter(loader))
                            loss, recon, kld = trainer._compute_loss(
                                bx.to(device), bc.to(device), bm.to(device),
                            )
                        print(f"  [{scaler}|s{seed}|f{fold}] ep {ep+1}/{n_epochs} "
                              f"| loss={loss.item():.4f} recon={recon.item():.4f} "
                              f"kld={kld.item():.4f}")

                # Sample: z ~ N(0, I) with train-fold condition vectors, size = val fold
                n_syn = vc.size(0)
                idx = torch.randperm(ct.size(0))[:n_syn]
                ct_sampled = ct[idx].to(device)
                gx = trainer.sample(ct_sampled)

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

                syn_path = os.path.join(
                    gen_dir, f'{scaler}_{seed}_1_syn_fold{fold}.csv'
                )
                ds_keep.to_csv(syn_path, index=False)

                fm, fq, _, _ = fid_eval.analysis_univariate(dr_keep, ds_keep)
                fm['seed'] = seed; fm['fold'] = fold
                fq['seed'] = seed; fq['fold'] = fold
                fold_metrics.append(fm)
                fold_quantiles.append(fq)

            fid_eval.save_report(fold_metrics, fold_quantiles, exp_dir, scaler)
            print(f"[DONE] {scaler} | seed={seed} | dir={exp_dir}")


if __name__ == '__main__':
    main()
