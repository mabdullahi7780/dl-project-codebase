# ICONIP 2026 — Final Paper Bundle

Self-contained source + compiled PDF for the cross-country TB severity paper.

## Layout

```
final_bundle/
├── main.tex              # paper source
├── refs.bib              # bibliography
├── main.bbl              # pre-built bibliography (so bibtex is optional)
├── main.pdf              # compiled paper (25 pages)
├── README.md             # this file
├── tables/               # all \input-able tables
│   ├── headline_table.tex          # main cross-country results
│   ├── published_baselines.tex     # comparison to CheXzero / GroupDRO / IW / K24
│   ├── cohort_summary.tex          # per-country TB Portals breakdown
│   ├── ablation_fusion.tex         # per-rung Hybrid-mode ablation
│   ├── ablation_a1.tex / a2 / a3   # per-mode ablations for SG-ALP / DH / DT
│   ├── paired_bootstrap.tex        # paired-bootstrap deltas vs R1 MSE
│   ├── ablation_table.tex          # combined per-mode ablation grid
│   ├── *.csv                       # raw data behind the tables
└── figures/
    ├── fig_shift_decomposition.pdf # §3 empirical shift figure (NEW)
    ├── fig_cavity_attention.pdf    # §6.4 R4b attention 3x3 (NEW)
    ├── fig_alp_spatial.pdf         # §6.5 spatial ALP maps 3x2 (NEW)
    ├── example_cxrs/               # Fig.2 cohort example panel
    ├── fig_headline_bars.pdf       # headline + reference anchors
    ├── fig_waterfall.pdf           # per-rung Hybrid waterfall
    ├── fig_significance_forest.pdf # Bonferroni-Holm forest plot
    ├── fig_pearson.pdf             # Pearson heatmap
    ├── fig_slope_cal.pdf           # regression-slope per country
    ├── fig_per_severity.pdf        # MAE by severity band
    ├── fig_cavity_auc.pdf          # cavity AUC (CLS vs spatial head)
    ├── fig_conformal.pdf           # conformal coverage / width
    └── fig_scatter.pdf             # predicted-vs-truth scatter
```

## Recompile

Requires a working TeX install with the LNCS class (`llncs.cls`) and bibtex
style (`splncs04.bst`) — both ship with MiKTeX / TeX Live by default.

```bash
pdflatex -interaction=nonstopmode main.tex
bibtex main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
```

If you skip bibtex, the included `main.bbl` is used as-is.

## What's new since the previous submission

1. **Empirical shift-decomposition figure (§3, Fig.1)** with three real-data
   panels: ALP density per country, cavity prevalence per country, and
   regression-slope compression per country.
2. **Published-method baseline table (§6.3, Table 5)** comparing to CheXzero
   plug-in, GroupDRO, Importance-Weighted regression, and our local K24
   replication.
3. **Cohort + dataset table (Table 1) and example CXR panel (Fig.2)** in §5.1.
4. **Cavity attention overlay (Fig.5) and spatial ALP maps (Fig.6)** wired
   into §6.4 and §6.5 from real R4b / SG-ALP head outputs.
5. **Descriptive naming:** A1 → SG-ALP (single-head ALP), A2 → DH (dual-head),
   A3 → DT (dual-task), Fusion → Hybrid, applied throughout main + tables.
6. **K24 framing:** all numbers cited in K24 rows are our local 5-seed
   reproductions on a publicly reproducible 5{,}010-image manifest, with the
   K24 paper cited for method-family attribution. K24's published numerical
   table is not quoted in main text per the round-3 framing request.

## Headline numbers (recap)

| Method | Romania | Moldova | Kazakhstan |
|---|---:|---:|---:|
| Best published baseline (any family) | 22.82 | 26.37 | 25.25 |
| K24 DH family, our 5-seed replication | 20.11 | 30.68 | 21.35 |
| **Hybrid + R3 + R6 + R4b (ours)**    | **19.12** | **21.59** | **16.74** |

Beats every published baseline on every held-out country.
