"""Per-country cohort breakdown for the paper's dataset section."""
from __future__ import annotations
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "baseline_runs" / "agentic_runs" / "paper" / "cohort_demographics.csv"
OUT_DIR = ROOT / "iconips_Paper" / "tables"
OUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(SRC, dtype={"image_id": str, "patient_id": str, "country": str})
df["timika"] = df["alp_0_100"] + 40.0 * df["cavity"]

# Per-country
g = df.groupby("country")
summary = pd.DataFrame({
    "images":     g.size(),
    "patients":   g["patient_id"].nunique(),
    "alp_mean":   g["alp_0_100"].mean().round(1),
    "alp_std":    g["alp_0_100"].std().round(1),
    "cavity_rate": (g["cavity"].mean() * 100).round(1),
    "timika_mean": g["timika"].mean().round(1),
    "timika_std":  g["timika"].std().round(1),
})
# Order by total cohort: Georgia/Ukraine first, then our three LOCO countries, then small ones
order = ["Georgia", "Ukraine", "Belarus", "Moldova", "Kazakhstan", "Romania",
         "Azerbaijan", "India"]
summary = summary.reindex([c for c in order if c in summary.index])
total = pd.DataFrame({
    "images":      [len(df)],
    "patients":    [df["patient_id"].nunique()],
    "alp_mean":    [round(df["alp_0_100"].mean(), 1)],
    "alp_std":     [round(df["alp_0_100"].std(), 1)],
    "cavity_rate": [round(df["cavity"].mean() * 100, 1)],
    "timika_mean": [round(df["timika"].mean(), 1)],
    "timika_std":  [round(df["timika"].std(), 1)],
}, index=["TOTAL"])
final = pd.concat([summary, total])
print(final.to_string())

final.to_csv(OUT_DIR / "cohort_summary.csv")

# LaTeX
lines = [
    r"\begin{table}[t]",
    r"\centering",
    r"\caption{TB Portals cohort, restricted to the paper manifest (one image per patient unless noted, 5{,}010 total).",
    r"ALP is the radiologist-annotated affected-lung percentage; Timika $=$ ALP $+ 40\cdot\text{cavity}$.",
    r"Held-out splits in our LOCO experiments use the three highlighted countries (Moldova, Kazakhstan, Romania).}",
    r"\label{tab:cohort}",
    r"\small",
    r"\begin{tabular}{lrrrrrr}",
    r"\toprule",
    r"Country & Images & Patients & ALP $\mu/\sigma$ & Cavity \% & Timika $\mu/\sigma$ \\",
    r"\midrule",
]
for c in final.index:
    r = final.loc[c]
    star = r"$^\dagger$" if c in {"Moldova", "Kazakhstan", "Romania"} else ""
    name = c if c != "TOTAL" else r"\textbf{Total}"
    lines.append(
        f"{name}{star} & {int(r['images'])} & {int(r['patients'])} & "
        f"{r['alp_mean']:.1f}/{r['alp_std']:.1f} & {r['cavity_rate']:.1f} & "
        f"{r['timika_mean']:.1f}/{r['timika_std']:.1f} \\\\"
    )
lines += [r"\bottomrule", r"\end{tabular}",
          r"\\[2pt]{\small $^\dagger$ Held-out LOCO test country.}",
          r"\end{table}"]
(OUT_DIR / "cohort_summary.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"\nWrote {OUT_DIR / 'cohort_summary.csv'}")
print(f"Wrote {OUT_DIR / 'cohort_summary.tex'}")
