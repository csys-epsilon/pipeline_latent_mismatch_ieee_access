# Pipeline-Latent Mismatch — Reference Implementation

Reference code accompanying the IEEE Access submission *"The Pipeline-Latent
Mismatch: Tail-Preserving Preprocessing of High-Kurtosis Clinical Biomarkers
in Tabular Generative Models."*

All hyperparameters and protocol choices are aligned with the manuscript's
primary configuration. See Section III of the manuscript for the corresponding
descriptions.

---

## Repository structure

```
.
├── configs/
│   └── config.yaml              # Unified configuration (manuscript primary settings)
├── scripts/
│   ├── run_tabddpm_fidelity.py  # Train TabDDPM, generate, score fidelity
│   ├── run_cvae_fidelity.py     # Same protocol with CVAE
│   ├── run_tstr_utility.py      # TSTR utility (4 classifiers × All / Tail-P90, bootstrap CI)
│   ├── run_latent_kl.py         # Component-level Latent KL (Fig. 3A x-axis)
│   ├── run_nn_diagnostic.py     # 1-NN ratio memorization diagnostic
│   ├── run_nhanes_external.py   # NHANES external TSTR + Real Baseline
│   └── run_mixed_effects.py     # Mixed-effects analysis (Table VIII)
├── src/
│   ├── preprocessing/scalers.py # 5 scalers + head-tail Adaptive (Eqs. 1-7, Section III-B)
│   ├── models/tabddpm.py        # MLP-based denoising diffusion (Section III-C)
│   ├── models/cvae.py           # Conditional VAE (Section III-C secondary)
│   └── evaluation/
│       ├── fidelity.py          # KS / KL / SMD / Wasserstein / kurtosis preservation
│       ├── tstr_utility.py      # XGBoost + LightGBM + RF + LR, bootstrap CI
│       ├── nn_distance.py       # 1-NN ratio under matched preprocessing
│       ├── latent_kl.py         # Component-level Latent KL (KDE on 500-pt grid)
│       └── mixed_effects.py     # REML + LRT + Holm-corrected pairwise contrasts
└── requirements.txt
```

---

## Reproducing the main experiments

### 0. Data preparation

Place the 10-fold stratified split files of the Arirang cohort at
`<ROOT_DIR>/raw_data/real_folds/real_f{1..10}.csv`. Each fold CSV must contain
the 31 features used in the 7-HK configuration (24 non-HK + 7 HK), the two
targets `is_mi` / `is_stroke`, and any identifier columns listed under
`DROP_COLS`. Update `ROOT_DIR` in `configs/config.yaml` to your absolute path.

Approval and access for the Arirang cohort are governed by Wonju Severance
Christian Hospital IRB (No. CR325148). NHANES 2017-2018 data are publicly
available from the U.S. CDC.

### 1. Section IV-A and IV-C — Five-scaler comparison under TabDDPM

```bash
python scripts/run_tabddpm_fidelity.py --config configs/config.yaml
python scripts/run_tstr_utility.py     --config configs/config.yaml --model tabDDPM \
                                       --include-real-baseline
```

Outputs:
- `evaluation/fidelity/tabDDPM/Arirang_All_vars/<scaler>_base/...`
  per-seed metrics and synthetic CSVs
- `evaluation/utility/tabDDPM/internal_Arirang_All_vars/tstr_summary_with_ci.csv`
  bootstrap-CI summary for the manuscript tables

### 2. Component-level Latent KL (Fig. 3A x-axis)

```bash
python scripts/run_latent_kl.py --config configs/config.yaml
```

Produces a CSV of Latent KL values per (scaler × HK variable) computed under
kernel density estimation on a 500-point empirical-support grid.

### 3. Section IV-B — Adaptive α sweep

Edit `configs/config.yaml`:

```yaml
SCALERS: ['adaptive']
ENABLE_ALPHA_SWEEP: true
ALPHA_SWEEP: [0.0, 0.1, 0.3, 0.5, 0.7, 1.0]
```

then:

```bash
python scripts/run_tabddpm_fidelity.py --config configs/config.yaml
python scripts/run_tstr_utility.py     --config configs/config.yaml --model tabDDPM \
                                       --pattern 'adaptive_alpha/*'
```

### 4. Section IV-D — CVAE replication and memorization diagnostics

```bash
python scripts/run_cvae_fidelity.py    --config configs/config.yaml
python scripts/run_tstr_utility.py     --config configs/config.yaml --model cvae \
                                       --include-real-baseline

# Four-metric memorization sweep across both architectures
python scripts/run_nn_diagnostic.py    --config configs/config.yaml \
                                       --archs tabDDPM cvae
```

The NN-diagnostic script produces three CSVs in
`<ROOT_DIR>/evaluation/memorization_diagnostic/`:
- `memorization_RAW.csv`       — one row per cell (arch × scaler × seed × fold)
- `memorization_SUMMARY.csv`   — per (arch × scaler) with 95% CI
- `memorization_ARCH.csv`      — pooled per architecture (Section IV-D summary)

The CVAE capacity ablation (Appendix Table C3) is obtained by setting
`CVAE.hidden_dim: 128` in the config and re-running. The β reweighting sweep
is obtained by varying `CVAE.beta` over {0.05, 0.1, 0.5, 1.0} with
`CVAE.latent_dim: 32` per Section III-C.

### 5. Section IV-E — NHANES external evaluation

```bash
# Synthetic-trained, NHANES-tested
python scripts/run_nhanes_external.py --config configs/config.yaml --model tabDDPM
python scripts/run_nhanes_external.py --config configs/config.yaml --model cvae

# NHANES TRTR upper bound
python scripts/run_nhanes_external.py --config configs/config.yaml --real-baseline
```

Requires `<ROOT_DIR>/raw_data/real_nhanes_folds/real_f{1..K}.csv`. The
NHANES 18-HK subgroup analysis (n = 11 thick-tail variables) re-runs the
above commands after producing 18-HK synthetic data with the NHANES 18-HK
feature set substituted into `preprocessing.split_quantile_features`.

### 6. Section IV-E — Mixed-effects pairwise contrasts (Table VIII)

The mixed-effects analysis consumes per-scaler pooled univariate CSVs and a
per-variable TTR CSV. Two modes are supported:

**Mode A — from pooled fidelity CSVs:**

```bash
python scripts/run_mixed_effects.py \
    --scaler-csvs Standard=results/Standard_univariate_pooled_final.csv \
                  Quantile=results/Quantile_univariate_pooled_final.csv \
                  Robust=results/Robust_univariate_pooled_final.csv \
                  Power=results/Power_univariate_pooled_final.csv \
                  Adaptive=results/Adaptive_univariate_pooled_final.csv \
    --ttr-csv results/ttr_per_variable.csv \
    --hk-variables results/hk_18_variables.txt \
    --out-dir results/mixed_effects \
    --reference Quantile --target Adaptive
```

**Mode B — from pre-built long-format CSV (columns
`[Feature, scaler, TTR, preservation_pp]`):**

```bash
python scripts/run_mixed_effects.py \
    --input  results/preservation_long.csv \
    --out-dir results/mixed_effects \
    --reference Quantile --target Adaptive
```

Outputs:
- `mixed_effects_results.txt`    — LRT, ICC, full model summary, Adaptive contrasts
- `pairwise_contrasts_full.csv`  — all 10 pairwise contrasts (Holm-corrected)
- `Adaptive_centric_contrasts.csv` — the 4 contrasts reported in Table VIII
- `mixed_effects_input.csv`      — audit copy of the long-format data fed to the fit

Requires `statsmodels >= 0.14`.

### 7. Section III-E — Three-stage progressive ablation (Non-HK / 3-HK / 7-HK)

Edit `preprocessing.split_quantile_features` in the config:

- Stage 0 (Non-HK):  `[]`
- Stage 1 (3-HK):    `[crtn_s, ins_s, tbil_s]`
- Stage 2 (7-HK):    the full list (default)

Re-run Steps 1 and 5 after each edit. Outputs are placed under separate
`experiment_condition` sub-folders if `PATHS.experiment_condition` is
updated correspondingly (`Arirang_Non_HK`, `Arirang_3_HK`, `Arirang_All_vars`).

### 8. Section III-E — Partition threshold sensitivity (P95 / P97.5 / P99)

Switch `base_tail_quantile` and proportionally adjust `mix_low_quantile` /
`mix_high_quantile` to keep the transition zone centered at the partition:

| Partition | base_tail_quantile | mix_low | mix_high |
|-----------|--------------------|---------|----------|
| P95       | 0.95               | 0.925   | 0.975    |
| P97.5     | 0.975              | 0.95    | 0.99     |
| P99       | 0.99               | 0.975   | 0.995    |

Then re-run Step 1 with `SCALERS: ['adaptive']` (this experiment is
fidelity-only per the manuscript).

---

## Configuration notes

A single config file controls all runs. The primary settings — scaler list,
five seeds × 10 folds, TabDDPM 512-512 / 200 steps / 2000 epochs / batch 512,
CVAE 256-hidden / latent-64 / batch 512 / 2000 epochs / β=0.01 — match
Manuscript v2.3 Section III. The capacity-ablation and β-sweep variants are
specified in the comments inside `configs/config.yaml`.

**Training-time configuration matches the manuscript's reported runs:**
- Both architectures apply a **2.5× reconstruction weight** to samples at or
  above the P97.5 threshold of any HK variable (configurable via
  `TABDDPM.tail_weight` and `CVAE.tail_weight`; set to 1.0 to disable). The
  weighting ensures adequate gradient signal in the heavy-tail regime where
  cardiovascular events concentrate, and is applied identically across every
  scaler condition — so relative scaler comparisons (the manuscript's
  central results) are unaffected by this choice.
- No early stopping; both models train the full 2000 epochs as specified.
- No gradient clipping, no learning-rate scheduler, no exponential moving averages.
- No SMOTE / oversampling / synthetic augmentation in the fidelity pipeline.

**Numerical guards (documented):**
- `Power` (Yeo-Johnson) inverse: latent clipped to [-5, +5] before
  `inverse_transform`. This is required to reproduce the manuscript's
  reported HK-mean kurtosis preservation of 82.4% (Table II); without it the
  closed-form 1/λ inverse overflows for a non-negligible fraction of samples
  and the result becomes undefined. The clip covers ≈99.99994% of N(0,1)
  support and does not affect non-tail values.
- `Adaptive` inverse: Newton-style refinement with bounded step size and
  finite-value guards inside the iteration loop. This is an algorithm-level
  safeguard for numerical stability, not a tail-quality intervention.

**TSTR classifier configuration:**
- Each classifier uses its library defaults. The only deviation is enabling
  the standard imbalance flag (`scale_pos_weight` for XGBoost, `is_unbalance`
  for LightGBM, `class_weight='balanced'` for RandomForest and
  LogisticRegression) because the positive prevalence is below 6%.
  LogisticRegression also raises `max_iter` from 100 to 1000 to ensure
  convergence — no other hyperparameter is tuned.
- The same classifier configuration is used across all scalers and the
  Real Baseline.

---

## License and citation

See LICENSE and CITATION.cff (to be added) for citation details.
The Arirang cohort data are not redistributed with this repository. NHANES
data are downloadable from https://www.cdc.gov/nchs/nhanes/.
