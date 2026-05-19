"""
Run TSTR (Train-on-Synthetic, Test-on-Real) utility evaluation.

Iterates over the scaler subfolders produced by run_tabddpm_fidelity.py or
run_cvae_fidelity.py, computes utility metrics for the four classifiers
on All and Tail-P90 subsets, and saves per-row results plus bootstrap-CI
aggregates.

Usage
-----
    python scripts/run_tstr_utility.py --config configs/config.yaml --model tabDDPM
    python scripts/run_tstr_utility.py --config configs/config.yaml --model cvae

By default, every <scaler>_base directory under the fidelity tree is
evaluated. To evaluate an Adaptive alpha sweep instead, pass:
    --pattern 'adaptive_alpha/*'
"""

import argparse
import glob
import os
import sys
import yaml

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.evaluation.tstr_utility import TSTRUtilityEvaluator


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--model',  choices=['tabDDPM', 'cvae'], default='tabDDPM')
    ap.add_argument('--pattern', default=None,
                    help="Optional glob pattern under fidelity_dir "
                         "(default: '*_base')")
    ap.add_argument('--include-real-baseline', action='store_true',
                    help="Also compute the TRTR Real Baseline")
    args = ap.parse_args()

    with open(args.config, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    paths = cfg.get('PATHS', {})
    root  = cfg['ROOT_DIR']
    exp   = paths.get('experiment_condition', 'Arirang_All_vars')
    fid_model = 'cvae' if args.model == 'cvae' else paths.get('fidelity_model', 'tabDDPM')
    util_model = fid_model
    util_int_prefix = paths.get('utility_internal_prefix', 'internal')

    def _p(*parts):
        return os.path.join(root, *[p.replace('/', os.sep) for p in parts])

    fold_dir     = _p(paths.get('real_folds_dir', 'raw_data/real_folds'))
    fidelity_dir = _p(paths.get('fidelity_root', 'evaluation/fidelity'),
                      fid_model, exp)
    out_dir      = _p(paths.get('utility_root',  'evaluation/utility'),
                      util_model, f'{util_int_prefix}_{exp}')

    pattern = args.pattern or '*_base'
    scaler_dirs = sorted(glob.glob(os.path.join(fidelity_dir, pattern)))
    if not scaler_dirs:
        print(f"[TSTR] no scaler subfolders matched: {fidelity_dir}/{pattern}")
        sys.exit(1)

    evaluator = TSTRUtilityEvaluator(cfg, fold_dir, fidelity_dir, out_dir)
    print(f"[TSTR] model={args.model}  fidelity_dir={fidelity_dir}")
    print(f"[TSTR] matched {len(scaler_dirs)} scaler folders: "
          f"{[os.path.basename(d) for d in scaler_dirs]}")

    all_rows = []
    for d in scaler_dirs:
        tag = os.path.relpath(d, fidelity_dir).replace(os.sep, '__')
        # Pattern of generated CSVs produced by run_*_fidelity.py:
        #   <scaler_dir>/<model>_<scaler>_Seed<seed>/generated_data/<scaler>_<seed>_1_syn_fold<fold>.csv
        rel_pattern = os.path.relpath(d, fidelity_dir)
        df = evaluator.evaluate_scaler(scaler_tag=tag, gen_data_pattern=rel_pattern)
        if not df.empty:
            all_rows.append(df)
            df.to_csv(os.path.join(out_dir, f'tstr_per_fold_{tag}.csv'), index=False)

    if args.include_real_baseline:
        df_real = evaluator.evaluate_real_baseline()
        if not df_real.empty:
            all_rows.append(df_real)
            df_real.to_csv(os.path.join(out_dir, 'tstr_per_fold_RealBaseline.csv'),
                           index=False)

    if not all_rows:
        print("[TSTR] no results produced.")
        return

    combined = pd.concat(all_rows, ignore_index=True)
    combined.to_csv(os.path.join(out_dir, 'tstr_per_fold_combined.csv'), index=False)
    evaluator.aggregate_with_ci(
        combined, os.path.join(out_dir, 'tstr_summary_with_ci.csv')
    )
    print(f"[TSTR] done. outputs in: {out_dir}")


if __name__ == '__main__':
    main()
