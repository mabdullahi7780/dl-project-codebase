"""Build the complete TB Portals manifest from all three releases.

Kantipudi et al. (JIIM 2024) used the full TB Portals dataset (~18,900 CXR
images across all released).  This script reproduces that dataset by combining:
  - August 2023 Archive release  (E:\\Archive\\)
  - March 2025 release           (E:\\March2025\\)
  - March 2026 release           (E:\\March2026\\)

Each release is CUMULATIVE: March 2026 contains ~95% of March 2025 studies,
which in turn contains ~99% of August 2023 studies.  The union of all three
releases is ~19,066 unique imaging studies.

Linking strategy
----------------
August 2023
  The CXR metadata CSV (TB_Portals_CXRs_August_2023.csv) has a
  ``series_instance_content_url`` column that encodes the exact DICOM path
  inside the Archive zip.  Mapping: imagingstudy_id -> zip entry path.

March 2025 / 2026 (new studies not in August 2023)
  These CXR CSVs lack ``series_instance_content_url``.  The only key shared
  between the metadata CSVs and the DICOM zip entry paths is ``patient_id``
  (= the patient UUID = first directory component of each zip entry).
  We build an index: patient_uuid -> {dicom_study_uid -> (zip_path, entry)}
  then match on patient_uuid.  For the minority of patients with multiple
  new studies, we pick the study whose first DICOM's StudyDate (from the
  DICOM header) best matches the annotation's imaging_date field, falling
  back to alphabetical order if dates are unavailable.

Output
------
A single CSV with columns:
  image_id, image_path, patient_id, country, alp_0_100, cavity

  image_path uses the !DICOM! zip-reference format understood by
  src/data/tbportals_dataset.py (e.g.
  "E:\\Archive\\...zip!DICOM!GlobalBucket/uuid/study/series/inst.dcm").

Usage::

    python scripts/build_full_manifest.py \\
        --out local_work/data/processed/tbportals_manifest_full.csv

    # then re-export to PNG for Kaggle:
    python scripts/export_kaggle_dataset.py \\
        --manifest local_work/data/processed/tbportals_manifest_full.csv \\
        --out-dir  local_work/kaggle_export_full
"""

from __future__ import annotations

import argparse
import csv
import io
import sys
import zipfile
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

try:
    import pydicom
    _PYDICOM = True
except ImportError:
    _PYDICOM = False
    print("[warn] pydicom not found — will skip StudyDate disambiguation for multi-study patients")

# ── Paths ─────────────────────────────────────────────────────────────────────
ARCHIVE_DIR  = Path("E:/Archive")
MARCH25_DIR  = Path("E:/March2025")
MARCH26_DIR  = Path("E:/March2026")

ARCHIVE_DATA_ZIP  = ARCHIVE_DIR / "TB_Portals_Published_Imaging_Data_August_2023.zip"
ARCHIVE_CXR_ZIP   = ARCHIVE_DIR / "TB_Portals_Published_CXRs_August_2023_Amended.zip"
MARCH25_DATA_ZIP  = MARCH25_DIR / "TB_Portals_Published_CXR_Data_March_2025.zip"
MARCH26_DATA_ZIP  = MARCH26_DIR / "TB_Portals_CXR_March_2026.zip"
MARCH25_DCM_DIR   = MARCH25_DIR / "TB_Portals_CXRs_March_2025"
MARCH26_DCM_DIR   = MARCH26_DIR / "TB_Portals_CXRs_March_2026"

_TRUE_TOKENS = {"1", "true", "yes", "y", "t", "present", "positive"}
_ZIP_SEP = "!DICOM!"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _read_csv_from_zip(zip_path: Path, name_fragment: str) -> list[dict]:
    with zipfile.ZipFile(zip_path) as z:
        entry = next(n for n in z.namelist() if name_fragment in n)
        with z.open(entry) as f:
            reader = csv.DictReader(io.TextIOWrapper(f, encoding="utf-8-sig"))
            return list(reader)


def _coerce_alp(val: str) -> float | None:
    try:
        v = float(val)
        return v if 0.0 <= v <= 100.0 else None
    except (TypeError, ValueError):
        return None


def _coerce_cavity(val: str) -> int | None:
    if val is None:
        return None
    s = str(val).strip().lower()
    if s in _TRUE_TOKENS:
        return 1
    try:
        return 1 if int(float(s)) > 0 else 0
    except (ValueError, TypeError):
        return None


def _encode_zip_ref(zip_path: str, entry: str) -> str:
    return zip_path + _ZIP_SEP + entry


def _read_dicom_study_date(zip_path: str, entry_name: str) -> str:
    """Return DICOM StudyDate (YYYYMMDD) or '' if unavailable."""
    if not _PYDICOM:
        return ""
    try:
        with zipfile.ZipFile(zip_path) as z:
            with z.open(entry_name) as f:
                data = f.read(8192)
        ds = pydicom.dcmread(io.BytesIO(data), stop_before_pixels=True)
        return str(getattr(ds, "StudyDate", "")).strip()
    except Exception:
        return ""


# ── Step 1: Load all annotation data ─────────────────────────────────────────

def load_annotations() -> dict[str, dict]:
    """Return imagingstudy_id -> {alp, cavity, imaging_date, modality}."""
    ann: dict[str, dict] = {}

    def _ingest(rows: list[dict], release: str) -> None:
        per_study: dict[str, list] = defaultdict(list)
        for row in rows:
            sid = row.get("imagingstudy_id", "").strip()
            if not sid:
                continue
            per_study[sid].append(row)

        for sid, srows in per_study.items():
            # overall ALP: take first non-null value across sextant rows
            alp = None
            for r in srows:
                alp = _coerce_alp(r.get("overallpercentofabnormalvolume", ""))
                if alp is not None:
                    break
            if alp is None:
                continue

            # cavity: any sextant with any cavity count > 0
            cav_cols = ["smallcavities", "mediumcavities", "largecavities"]
            cavity = 0
            for r in srows:
                for c in cav_cols:
                    try:
                        if int(float(r.get(c, 0) or 0)) > 0:
                            cavity = 1
                    except (ValueError, TypeError):
                        pass

            modality = srows[0].get("series_modality_cd", "").strip()
            imaging_date = srows[0].get("imaging_date", "").strip()

            # prefer most recent release annotation (overwrite earlier)
            ann[sid] = {
                "alp": alp,
                "cavity": cavity,
                "modality": modality,
                "imaging_date": imaging_date,
                "release": release,
            }

    print("[manifest] loading annotations…")
    _ingest(_read_csv_from_zip(ARCHIVE_DATA_ZIP, "TB_Portals_CXR_Manual_Annotations_August_2023.csv"), "aug2023")
    _ingest(_read_csv_from_zip(MARCH25_DATA_ZIP, "TB_Portals_CXR_Manual_Annotations_March_2025.csv"), "mar2025")
    _ingest(_read_csv_from_zip(MARCH26_DATA_ZIP, "TB_Portals_CXR_Manual_Annotations_March_2026.csv"), "mar2026")
    print(f"[manifest]   {len(ann)} unique annotated studies (after dedup)")
    return ann


# ── Step 2: Load CXR metadata (patient_id, country, imagingstudy_id) ─────────

def load_metadata() -> dict[str, dict]:
    """Return imagingstudy_id -> {patient_id, country}."""
    meta: dict[str, dict] = {}

    def _ingest(rows: list[dict]) -> None:
        for row in rows:
            sid = row.get("imagingstudy_id", "").strip()
            if not sid:
                continue
            meta[sid] = {
                "patient_id": row.get("patient_id", "").strip(),
                "country": row.get("country", "").strip(),
            }

    print("[manifest] loading CXR metadata…")
    _ingest(_read_csv_from_zip(ARCHIVE_DATA_ZIP, "TB_Portals_CXRs_August_2023.csv"))
    _ingest(_read_csv_from_zip(MARCH25_DATA_ZIP, "TB_Portals_CXRs_March_2025.csv"))
    _ingest(_read_csv_from_zip(MARCH26_DATA_ZIP, "TB_Portals_CXRs_March_2026.csv"))
    print(f"[manifest]   {len(meta)} unique studies in metadata")
    return meta


# ── Step 3a: August 2023 path index ──────────────────────────────────────────

def build_aug2023_index() -> dict[str, str]:
    """imagingstudy_id -> zip-ref path (for Archive zip studies)."""
    print("[manifest] indexing August 2023 Archive paths…")
    idx: dict[str, str] = {}
    rows = _read_csv_from_zip(ARCHIVE_DATA_ZIP, "TB_Portals_CXRs_August_2023.csv")
    archive_zip = str(ARCHIVE_CXR_ZIP)
    for row in rows:
        sid = row.get("imagingstudy_id", "").strip()
        url = row.get("series_instance_content_url", "").strip()
        if sid and url:
            # Archive zip has GlobalBucket/ prefix
            entry = "GlobalBucket/" + url
            idx[sid] = _encode_zip_ref(archive_zip, entry)
    print(f"[manifest]   {len(idx)} Aug2023 paths indexed")
    return idx


# ── Step 3b: March 2025/2026 patient→study index ──────────────────────────────

def build_march_patient_index() -> dict[str, dict[str, tuple[str, str]]]:
    """patient_uuid -> {dicom_study_uid -> (zip_path, representative_entry)}."""
    print("[manifest] scanning March 2025/2026 zip entries (patient index)…")
    # patient_uuid -> {dicom_study_uid -> (zip_path, entry_name)}
    idx: dict[str, dict[str, tuple[str, str]]] = defaultdict(dict)

    def _scan(dcm_dir: Path) -> None:
        zips = sorted(dcm_dir.glob("*.zip"))
        print(f"[manifest]   {dcm_dir.name}: {len(zips)} part-zip(s)")
        for zip_path in zips:
            with zipfile.ZipFile(zip_path) as z:
                for entry in z.namelist():
                    if not entry.endswith(".dcm"):
                        continue
                    parts = entry.split("/")
                    if len(parts) < 3:
                        continue
                    patient_uuid = parts[0]
                    study_uid = parts[1]
                    if study_uid not in idx[patient_uuid]:
                        idx[patient_uuid][study_uid] = (str(zip_path), entry)

    _scan(MARCH25_DCM_DIR)
    _scan(MARCH26_DCM_DIR)
    print(f"[manifest]   {len(idx)} unique patients indexed from March zips")
    return idx


# ── Step 4: Resolve DICOM path for each annotated study ──────────────────────

def resolve_paths(
    ann: dict[str, dict],
    meta: dict[str, dict],
    aug_idx: dict[str, str],
    march_idx: dict[str, dict[str, tuple[str, str]]],
) -> list[dict]:
    print("[manifest] resolving DICOM paths…")
    rows = []
    n_aug = n_march = n_missing = n_no_meta = n_bad_modality = 0

    for sid, ainfo in ann.items():
        # Only keep CXR modality
        if ainfo["modality"] not in ("XC", "CR", "DX", ""):
            n_bad_modality += 1
            continue

        if sid not in meta:
            n_no_meta += 1
            continue

        minfo = meta[sid]
        patient_id = minfo["patient_id"]
        country = minfo["country"]
        if not patient_id or not country:
            n_no_meta += 1
            continue

        # Try August 2023 index first
        if sid in aug_idx:
            image_path = aug_idx[sid]
            n_aug += 1
        else:
            # Find in March patient index
            patient_studies = march_idx.get(patient_id, {})
            if not patient_studies:
                n_missing += 1
                continue

            if len(patient_studies) == 1:
                study_uid, (zip_path, entry) = next(iter(patient_studies.items()))
                image_path = _encode_zip_ref(zip_path, entry)
            else:
                # Multiple DICOM studies for this patient — try StudyDate disambiguation
                imaging_date = ainfo.get("imaging_date", "").strip()
                best_entry = None
                best_score = -1
                for study_uid, (zip_path, entry) in sorted(patient_studies.items()):
                    if imaging_date and _PYDICOM:
                        dcm_date = _read_dicom_study_date(zip_path, entry)
                        # imaging_date in TB Portals is an integer day offset; DICOM
                        # StudyDate is YYYYMMDD.  If imaging_date is a valid YYYYMMDD
                        # string, use direct match; otherwise fall back to order.
                        if dcm_date and imaging_date.replace("-", "") == dcm_date:
                            best_entry = (zip_path, entry)
                            break
                        score = int(dcm_date) if dcm_date.isdigit() else 0
                        if score > best_score:
                            best_score = score
                            best_entry = (zip_path, entry)
                    else:
                        # fallback: alphabetically first study_uid
                        if best_entry is None:
                            best_entry = (zip_path, entry)
                if best_entry is None:
                    n_missing += 1
                    continue
                image_path = _encode_zip_ref(best_entry[0], best_entry[1])
            n_march += 1

        rows.append({
            "image_id":   sid,
            "image_path": image_path,
            "patient_id": patient_id,
            "country":    country,
            "alp_0_100":  round(ainfo["alp"], 4),
            "cavity":     ainfo["cavity"],
        })

    print(f"[manifest]   aug2023 paths:  {n_aug}")
    print(f"[manifest]   march paths:    {n_march}")
    print(f"[manifest]   missing DICOM:  {n_missing}")
    print(f"[manifest]   no metadata:    {n_no_meta}")
    print(f"[manifest]   bad modality:   {n_bad_modality}")
    return rows


# ── Step 5: Summary ───────────────────────────────────────────────────────────

def _summarize(df: pd.DataFrame) -> None:
    print("\n[manifest] Per-country summary:")
    g = df.groupby("country").agg(
        n=("image_id", "size"),
        alp_mean=("alp_0_100", "mean"),
        cavity_rate=("cavity", "mean"),
    ).sort_values("n", ascending=False)
    print(g.to_string())
    print(f"\n[manifest] Total: {len(df)} images, {df['patient_id'].nunique()} patients, "
          f"{df['country'].nunique()} countries")


# ── Main ──────────────────────────────────────────────────────────────────────

def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Build full TB Portals manifest from all releases.")
    p.add_argument("--out", default="local_work/data/processed/tbportals_manifest_full.csv",
                   help="Output CSV path.")
    args = p.parse_args(argv)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ann      = load_annotations()
    meta     = load_metadata()
    aug_idx  = build_aug2023_index()
    march_idx = build_march_patient_index()

    rows = resolve_paths(ann, meta, aug_idx, march_idx)

    df = pd.DataFrame(rows, columns=["image_id", "image_path", "patient_id", "country", "alp_0_100", "cavity"])
    df = df.drop_duplicates(subset=["image_id"]).reset_index(drop=True)

    _summarize(df)

    df.to_csv(out_path, index=False)
    print(f"\n[manifest] Written: {out_path}  ({len(df)} rows)")
    print("\nNext step:")
    print(f"  python scripts/export_kaggle_dataset.py --manifest {out_path} --out-dir local_work/kaggle_export_full")


if __name__ == "__main__":
    main()
