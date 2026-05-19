"""
Component-level Latent KL evaluation.

Applies each of the five scalers to the full Arirang cohort and computes the
Latent KL divergence between the transformed output and N(0, 1) for the seven
HK variables. Reproduces the x-axis of Manuscript v2.3 Fig. 3A.

Usage
-----
    python scripts/run_latent_kl.py --config configs/config.yaml
"""

import argparse
import os
import sys
import yaml

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.preprocessing.scalers import DataProcess
from src.evaluation.latent_kl import latent_kl_table


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--out', default='latent_kl_summary.csv')
    args = ap.parse_args()

    with open(args.config, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    root = cfg['ROOT_DIR']
    paths = cfg.get('PATHS', {})
    fold_dir = os.path.join(
        root, paths.get('real_folds_dir', 'raw_data/real_folds').replace('/', os.sep)
    )
    n_folds = int(cfg.get('KFOLD_N', 10))
    hk_features = cfg.get('preprocessing', {}).get('split_quantile_features', [])
    scalers = cfg.get('SCALERS', ['standard', 'quantile', 'robust', 'power', 'adaptive'])

    # Aggregate all folds (Latent KL is computed on the full distribution, not per fold)
    df = pd.concat([
        pd.read_csv(os.path.join(fold_dir, f'real_f{i}.csv'))
        for i in range(1, n_folds + 1)
    ], ignore_index=True)

    rows = []
    for scaler in scalers:
        dp = DataProcess(cfg)
        dp.set_global_scaler(df)
        # Forward-transform on the full set (no Adaptive split-fit needed when
        # we evaluate on the same data); use preprocess_train_valid with
        # train_df = val_df = df so adaptive scalers are fitted.
        _, _, _, vx, _, num_f, _ = dp.preprocess_train_valid(df, df, scaler)
        # Map tensor back to a dataframe with column names
        import torch
        scaled = pd.DataFrame(vx.numpy() if isinstance(vx, torch.Tensor) else vx,
                              columns=num_f)
        tbl = latent_kl_table(scaled, hk_features)
        tbl.insert(0, 'Scaler', scaler)
        rows.append(tbl)
        print(f"[Latent KL] {scaler}: HK-mean = {tbl['LatentKL'].mean():.4f}")

    out_df = pd.concat(rows, ignore_index=True)
    out_path = os.path.join(root, args.out)
    out_df.to_csv(out_path, index=False)
    print(f"\n[Latent KL] saved: {out_path}")

    # HK-mean summary
    hk_mean = (
        out_df.groupby('Scaler', as_index=False)['LatentKL']
              .mean()
              .sort_values('LatentKL')
    )
    print("\nHK-mean Latent KL by scaler:")
    print(hk_mean.to_string(index=False))


if __name__ == '__main__':
    main()
