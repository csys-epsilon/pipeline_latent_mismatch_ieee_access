"""
NHANES external TSTR evaluation.

Trains classifiers on Arirang synthetic data (per scaler × seed × fold) and
evaluates on the NHANES 2017-2018 cohort. Implements the external-validation
protocol described in Manuscript v2.3 Section IV-E.

NHANES Real Baseline (TRTR-on-NHANES) is computed separately by training and
testing on NHANES folds (use --real-baseline).

Usage
-----
    python scripts/run_nhanes_external.py --config configs/config.yaml --model tabDDPM
    python scripts/run_nhanes_external.py --config configs/config.yaml --model cvae
    python scripts/run_nhanes_external.py --config configs/config.yaml --real-baseline
"""

import argparse
import glob
import os
import sys
import yaml

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.evaluation.tstr_utility import TSTRUtilityEvaluator


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--model', choices=['tabDDPM', 'cvae'], default='tabDDPM')
    ap.add_argument('--pattern', default='*_base')
    ap.add_argument('--real-baseline', action='store_true',
                    help="Compute TRTR-on-NHANES baseline (no synthetic).")
    args = ap.parse_args()

    with open(args.config, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    root = cfg['ROOT_DIR']
    paths = cfg.get('PATHS', {})

    def _p(*parts):
        return os.path.join(root, *[p.replace('/', os.sep) for p in parts])

    nhanes_dir = _p(paths.get('nhanes_folds_dir', 'raw_data/real_nhanes_folds'))
    arirang_fid_dir = _p(
        paths.get('fidelity_root', 'evaluation/fidelity'),
        ('cvae' if args.model == 'cvae' else paths.get('fidelity_model', 'tabDDPM')),
        paths.get('experiment_condition', 'Arirang_All_vars'),
    )
    util_model = 'cvae' if args.model == 'cvae' else paths.get('utility_model', 'tabDDPM')
    util_ext_prefix = paths.get('utility_external_prefix', 'external_NHANES')
    out_dir = _p(
        paths.get('utility_root', 'evaluation/utility'),
        util_model,
        f'{util_ext_prefix}_{paths.get("experiment_condition", "Arirang_All_vars").replace("Arirang_", "")}',
    )
    os.makedirs(out_dir, exist_ok=True)

    if not os.path.isdir(nhanes_dir):
        print(f"[NHANES] missing NHANES fold directory: {nhanes_dir}")
        sys.exit(1)

    # The TSTRUtilityEvaluator already supports overriding fold_dir; we pass
    # the NHANES directory as fold_dir, and the Arirang synthetic folder as
    # fidelity_dir. The classifier trains on synthetic (Arirang) and tests on
    # the NHANES "fold" (we use the union of all NHANES folds as a single
    # test cohort, per Section IV-E protocol).
    evaluator = TSTRUtilityEvaluator(
        cfg, fold_dir=nhanes_dir,
        fidelity_dir=arirang_fid_dir, output_dir=out_dir,
    )

    if args.real_baseline:
        df_real = evaluator.evaluate_real_baseline()
        df_real.to_csv(os.path.join(out_dir, 'nhanes_real_baseline_per_fold.csv'),
                       index=False)
        evaluator.aggregate_with_ci(
            df_real, os.path.join(out_dir, 'nhanes_real_baseline_with_ci.csv')
        )
        print("[NHANES] real-baseline done.")
        return

    scaler_dirs = sorted(glob.glob(os.path.join(arirang_fid_dir, args.pattern)))
    if not scaler_dirs:
        print(f"[NHANES] no scaler folders under {arirang_fid_dir}/{args.pattern}")
        sys.exit(1)

    all_rows = []
    for d in scaler_dirs:
        tag = os.path.relpath(d, arirang_fid_dir).replace(os.sep, '__')
        rel_pattern = os.path.relpath(d, arirang_fid_dir)
        df = evaluator.evaluate_scaler(scaler_tag=tag, gen_data_pattern=rel_pattern)
        if not df.empty:
            all_rows.append(df)
            df.to_csv(os.path.join(out_dir, f'nhanes_tstr_per_fold_{tag}.csv'),
                      index=False)

    if not all_rows:
        print("[NHANES] no results produced.")
        return

    combined = pd.concat(all_rows, ignore_index=True)
    combined.to_csv(os.path.join(out_dir, 'nhanes_tstr_per_fold_combined.csv'),
                    index=False)
    evaluator.aggregate_with_ci(
        combined, os.path.join(out_dir, 'nhanes_tstr_summary_with_ci.csv')
    )
    print(f"[NHANES] done. outputs in: {out_dir}")


if __name__ == '__main__':
    main()
