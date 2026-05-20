"""
Mixed-effects analysis of kurtosis preservation across scalers (Table VIII).

Two input modes are supported:

(A) `--scaler-csvs` : pooled per-scaler univariate summary CSVs from the
    fidelity pipeline. The script concatenates them, computes the
    preservation rate, optionally filters to the HK variable set, and joins
    TTR from `--ttr-csv`.

(B) `--input` : a pre-built long-format CSV with columns
        [Feature, scaler, TTR, preservation_pp]

After fitting the REML model, the script writes:

    mixed_effects_results.txt    : LRT, ICC, full model summary, Adaptive contrasts
    pairwise_contrasts_full.csv  : all 10 pairwise contrasts
    mixed_effects_input.csv      : the long-format data used for the fit (audit)

Usage
-----
Mode A:
    python scripts/run_mixed_effects.py \\
        --scaler-csvs Standard=results/Standard_pooled.csv \\
                      Quantile=results/Quantile_pooled.csv \\
                      Robust=results/Robust_pooled.csv \\
                      Power=results/Power_pooled.csv \\
                      Adaptive=results/Adaptive_pooled.csv \\
        --ttr-csv results/ttr_per_variable.csv \\
        --hk-variables results/hk_18_variables.txt \\
        --out-dir results/mixed_effects \\
        --reference Quantile --target Adaptive

Mode B:
    python scripts/run_mixed_effects.py \\
        --input results/preservation_long.csv \\
        --out-dir results/mixed_effects \\
        --reference Quantile --target Adaptive
"""

import argparse
import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.evaluation.mixed_effects import (
    build_long_format,
    attach_ttr,
    fit_mixedlm,
    pairwise_contrasts,
    filter_target_centric,
)


def _parse_scaler_csvs(items):
    out = {}
    for it in items or []:
        if '=' not in it:
            raise ValueError(
                f"--scaler-csvs entries must be NAME=PATH, got: {it!r}"
            )
        name, path = it.split('=', 1)
        out[name.strip()] = path.strip()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', default=None,
                    help="Mode B: pre-built long-format CSV "
                         "[Feature, scaler, TTR, preservation_pp]")
    ap.add_argument('--scaler-csvs', nargs='+', default=None,
                    help="Mode A: list of NAME=PATH pairs "
                         "(one per scaler's pooled univariate CSV)")
    ap.add_argument('--ttr-csv', default=None,
                    help="Mode A: per-variable TTR CSV with columns "
                         "[Feature, TTR]")
    ap.add_argument('--hk-variables', default=None,
                    help="Mode A: optional text file with one HK variable per line "
                         "(restricts the analysis to these features)")
    ap.add_argument('--kurt-real-col', default='Kurtosis_Real_Conv_Mean')
    ap.add_argument('--kurt-syn-col', default='Kurtosis_Syn_Conv_Mean')
    ap.add_argument('--reference', default='Quantile')
    ap.add_argument('--target', default='Adaptive')
    ap.add_argument('--out-dir', required=True)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # ---------- Build long-format data ----------
    if args.input is not None:
        df = pd.read_csv(args.input)
        needed = {'Feature', 'scaler', 'TTR', 'preservation_pp'}
        missing = needed - set(df.columns)
        if missing:
            raise ValueError(f"Input is missing required columns: {missing}")
        import numpy as np
        df['log_TTR'] = np.log(df['TTR'])
    else:
        scaler_map = _parse_scaler_csvs(args.scaler_csvs)
        if not scaler_map:
            raise ValueError("Provide either --input or --scaler-csvs")
        hk_list = None
        if args.hk_variables and os.path.exists(args.hk_variables):
            with open(args.hk_variables, 'r', encoding='utf-8') as fh:
                hk_list = [ln.strip() for ln in fh if ln.strip()]
        df = build_long_format(
            scaler_map, hk_variables=hk_list,
            kurt_real_col=args.kurt_real_col,
            kurt_syn_col=args.kurt_syn_col,
        )
        if args.ttr_csv is None:
            raise ValueError("Mode A requires --ttr-csv")
        df = attach_ttr(df, args.ttr_csv)

    audit_path = os.path.join(args.out_dir, 'mixed_effects_input.csv')
    df.to_csv(audit_path, index=False)
    print(f"[saved] audit input -> {audit_path}")

    # ---------- Fit ----------
    md_full, chi2, df_lrt, p_lrt, icc, var_re, var_res = fit_mixedlm(
        df,
        response='preservation_pp', predictor='log_TTR',
        scaler_col='scaler', variable_col='Feature',
        reference_scaler=args.reference,
    )
    print(f"\n[MixedLM] N = {len(df)}, groups = {df['Feature'].nunique()}")
    print(f"[MixedLM] LRT chi2({df_lrt}) = {chi2:.3f}, p = {p_lrt:.4f}")
    print(f"[MixedLM] ICC = {icc:.3f}  "
          f"(random var = {var_re:.2f}, residual var = {var_res:.2f})")
    print(md_full.summary().tables[1])

    # ---------- Per-scaler means ----------
    per_scaler = (
        df.groupby('scaler', observed=True)['preservation_pp']
          .agg(['mean', 'std', 'count'])
          .sort_values('mean', ascending=False)
    )
    print("\nPer-scaler mean kurtosis preservation (pp):")
    print(per_scaler.to_string())

    # ---------- Pairwise contrasts ----------
    scalers = list(df['scaler'].cat.categories) if hasattr(df['scaler'], 'cat') \
        else sorted(df['scaler'].unique())
    if args.reference not in scalers:
        scalers = [args.reference] + [s for s in scalers if s != args.reference]
    pw = pairwise_contrasts(md_full, scalers, reference_scaler=args.reference)

    pw_path = os.path.join(args.out_dir, 'pairwise_contrasts_full.csv')
    pw.to_csv(pw_path, index=False)
    print(f"\n[saved] all pairwise contrasts -> {pw_path}")
    print(pw.to_string(index=False))

    # ---------- Target-centric (e.g. Adaptive) ----------
    target_df = filter_target_centric(pw, target=args.target)
    target_path = os.path.join(args.out_dir, f'{args.target}_centric_contrasts.csv')
    target_df.to_csv(target_path, index=False)
    print(f"\n[saved] {args.target}-centric contrasts -> {target_path}")
    print(target_df.to_string(index=False))

    # ---------- Text summary ----------
    text_path = os.path.join(args.out_dir, 'mixed_effects_results.txt')
    with open(text_path, 'w', encoding='utf-8') as fh:
        fh.write("Mixed-Effects Analysis Results\n")
        fh.write("=" * 70 + "\n\n")
        fh.write(f"N = {len(df)} "
                 f"({df['Feature'].nunique()} variables × "
                 f"{df['scaler'].nunique()} scalers)\n\n")
        fh.write(f"LRT chi2({df_lrt}) = {chi2:.3f}, p = {p_lrt:.4f}\n")
        fh.write(f"ICC = {icc:.3f}\n")
        fh.write(f"Random intercept variance = {var_re:.2f}\n")
        fh.write(f"Residual variance         = {var_res:.2f}\n\n")
        fh.write("Per-scaler mean preservation (pp):\n")
        fh.write(per_scaler.to_string() + "\n\n")
        fh.write("Full model summary:\n")
        fh.write(str(md_full.summary()) + "\n\n")
        fh.write(f"{args.target}-centric contrasts (Holm-corrected):\n")
        fh.write(target_df.to_string(index=False) + "\n")
    print(f"\n[saved] text summary -> {text_path}")


if __name__ == '__main__':
    main()
