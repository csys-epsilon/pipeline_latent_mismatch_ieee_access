"""
Memorization diagnostics over (architecture × scaler × seed × fold) cells.

Sweeps TabDDPM and CVAE outputs, computes the four manuscript diagnostics
(1-NN ratio, DCR ratio, hit rate at P5, NNDR) per cell, and writes:

    memorization_RAW.csv       : one row per cell
    memorization_SUMMARY.csv   : per (architecture × scaler), 95% CI
    memorization_ARCH.csv      : pooled per architecture

Usage
-----
    python scripts/run_nn_diagnostic.py --config configs/config.yaml \\
                                        --archs tabDDPM cvae \\
                                        --out-dir results/memorization
"""

import argparse
import glob
import os
import re
import sys
import yaml
from itertools import product

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.evaluation.nn_distance import (
    compute_nn_diagnostics,
    aggregate_arch_scaler,
    aggregate_arch_level,
)


SYN_FILE_RE = re.compile(
    r'^(?P<scaler>[a-zA-Z]+)_(?P<seed>\d+)_\d+_syn_fold(?P<fold>\d+)\.csv$'
)


def _resolve_arch_paths(cfg, arch):
    """Return (fold_dir, fidelity_root_for_arch) for a given architecture."""
    root = cfg['ROOT_DIR']
    paths = cfg.get('PATHS', {})

    def _p(*parts):
        return os.path.join(root, *[p.replace('/', os.sep) for p in parts])

    fold_dir = _p(paths.get('real_folds_dir', 'raw_data/real_folds'))
    fid_root = paths.get('fidelity_root', 'evaluation/fidelity')
    fid_model = 'cvae' if arch.lower() == 'cvae' else paths.get('fidelity_model', 'tabDDPM')
    exp = paths.get('experiment_condition', 'Arirang_All_vars')
    fid_dir = _p(fid_root, fid_model, exp)
    return fold_dir, fid_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--archs', nargs='+', default=['tabDDPM', 'cvae'])
    ap.add_argument('--out-dir', default=None,
                    help='Directory for output CSVs. Defaults to '
                         '<ROOT_DIR>/evaluation/memorization_diagnostic')
    args = ap.parse_args()

    with open(args.config, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    n_folds = int(cfg.get('KFOLD_N', 10))
    seeds = cfg.get('SEEDS', [42, 52, 62, 72, 82])
    drop_cols = cfg.get('DROP_COLS', []) + cfg.get('COND_VARS', [])
    exclude = list(set(drop_cols + ['p_id_group']))

    if args.out_dir is None:
        args.out_dir = os.path.join(
            cfg['ROOT_DIR'].replace('/', os.sep),
            'evaluation', 'memorization_diagnostic',
        )
    os.makedirs(args.out_dir, exist_ok=True)

    all_rows = []
    for arch in args.archs:
        fold_dir, fid_dir = _resolve_arch_paths(cfg, arch)
        print(f"\n--- {arch} ---  fid_dir = {fid_dir}")
        if not os.path.isdir(fid_dir):
            print(f"  [skip] not a directory: {fid_dir}")
            continue

        syn_files = glob.glob(os.path.join(
            fid_dir, '*', '*', 'generated_data', '*_syn_fold*.csv'
        ))
        if not syn_files:
            print(f"  [skip] no synthetic CSVs under {fid_dir}")
            continue

        n_cells = 0
        for syn_path in syn_files:
            m = SYN_FILE_RE.match(os.path.basename(syn_path))
            if not m:
                continue
            scaler = m.group('scaler')
            seed   = int(m.group('seed'))
            fold   = int(m.group('fold'))

            test_path = os.path.join(fold_dir, f'real_f{fold}.csv')
            if not os.path.exists(test_path):
                continue

            try:
                test_df = pd.read_csv(test_path)
                train_df = pd.concat([
                    pd.read_csv(os.path.join(fold_dir, f'real_f{i}.csv'))
                    for i in range(1, n_folds + 1) if i != fold
                ], ignore_index=True)
                syn_df = pd.read_csv(syn_path)

                metrics = compute_nn_diagnostics(
                    train_df, test_df, syn_df, exclude_cols=exclude
                )
                metrics.update({
                    'arch': arch, 'scaler': scaler,
                    'seed': seed, 'fold': fold,
                    'syn_path': syn_path,
                })
                all_rows.append(metrics)
                n_cells += 1
            except Exception as exc:
                print(f"  [err] {arch}/{scaler}/s{seed}/f{fold}: {exc}")

        print(f"  collected {n_cells} cells for {arch}")

    if not all_rows:
        print("\n[NN] no cells produced; nothing to save.")
        return

    df_raw = pd.DataFrame(all_rows)
    raw_path = os.path.join(args.out_dir, 'memorization_RAW.csv')
    df_raw.to_csv(raw_path, index=False)
    print(f"\n[NN] raw cells   -> {raw_path}  ({len(df_raw)} rows)")

    df_scaler = aggregate_arch_scaler(df_raw)
    sum_path = os.path.join(args.out_dir, 'memorization_SUMMARY.csv')
    df_scaler.to_csv(sum_path, index=False)
    print(f"[NN] per (arch × scaler) -> {sum_path}")
    print(df_scaler.to_string(index=False))

    df_arch = aggregate_arch_level(df_raw)
    arch_path = os.path.join(args.out_dir, 'memorization_ARCH.csv')
    df_arch.to_csv(arch_path, index=False)
    print(f"\n[NN] pooled per arch     -> {arch_path}")
    print(df_arch.to_string(index=False))


if __name__ == '__main__':
    main()
