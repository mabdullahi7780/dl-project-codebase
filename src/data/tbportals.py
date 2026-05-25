"""TB Portals data adapter -> manifest contract.

This is the ONLY module that touches TB Portals' raw schema. Everything
downstream (dataset, training, eval) reads a single ``manifest.csv`` with the
fixed columns in :data:`MANIFEST_COLUMNS`.

TB Portals Published Imaging Data zip structure (August 2023 release)
----------------------------------------------------------------------
The zip contains two relevant CSVs that must be joined on ``imagingstudy_ID``:

  1. ``TB Portals CXRs.csv``
       Image metadata: ``series_instance_content_url`` (filepath relative to
       the CXR root), ``condition_id`` (patient), ``country``, ``imagingstudy_ID``.

  2. ``TB Portals Published CXR Manual Annotations.csv``
       Radiologist labels: ``overallpercentofabnormalvolume`` (ALP 0-100),
       ``cavitiespresent`` (binary), ``imagingstudy_ID``.

Use :func:`build_manifest_from_tbportals_zip` for the standard workflow.
Use :func:`build_manifest_from_raw` if you have a pre-joined CSV.

Manifest contract::

    image_id, image_path, patient_id, country, alp_0_100, cavity

CLI::

    python -m src.data.tbportals zip \\
        --zip E:\\Archive\\TB_Portals_Published_Imaging_Data_August_2023.zip \\
        --image-roots E:\\Archive E:\\March2025 E:\\March2026 \\
        --out data/processed/tbportals_manifest.csv

    python -m src.data.tbportals synth --out data/processed/synth --n 600
"""

from __future__ import annotations

import argparse
import os
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# TB Portals schema constants (August 2023 release)
# ---------------------------------------------------------------------------

# Substring match against CSV filenames inside the zip.
# Actual filenames (August 2023 release):
#   TB_Portals_CXRs_August_2023.csv
#   TB_Portals_CXR_Manual_Annotations_August_2023.csv
# _find_entry_in_zip normalises spaces/underscores before matching.
TBPORTALS_CXR_CSV = "CXRs"                    # unique: matches CXRs, not CXR_Manual*
TBPORTALS_ANNOTATIONS_CSV = "CXR_Manual_Annotations"  # unique: not CTs, not Qure
JOIN_KEY = "imagingstudy_id"   # all-lowercase in the August 2023 CSVs

# Column names in the merged (joined) table. Pass column_map={...} to any
# build function to override these without editing source.
COLUMN_MAP: dict[str, str] = {
    "image_id":   "series_instance_content_url",   # unique per image in CXRs file
    "image_path": "series_instance_content_url",   # relative path to the DICOM
    "patient_id": "condition_id",
    "country":    "country",
    "alp":        "overallpercentofabnormalvolume",  # overall-lung ALP, 0-100 scalar
    "cavity":     "cavitiespresent",                 # binary; 'Yes'/'No' or 1/0
}

MANIFEST_COLUMNS = ["image_id", "image_path", "patient_id", "country", "alp_0_100", "cavity"]

# Kantipudi et al. (JIIM 2024) held-out countries for the country-segregated test.
HELD_OUT_COUNTRIES = ["Romania", "Moldova", "Kazakhstan"]

_TRUE_TOKENS  = {"1", "true", "yes", "y", "t", "present", "positive"}
_FALSE_TOKENS = {"0", "false", "no", "n", "f", "absent", "negative", "", "nan", "none"}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _coerce_alp(values: pd.Series) -> pd.Series:
    """Coerce ALP to float in [0, 100]. Auto-detects and rescales 0-1 encoding."""
    alp = pd.to_numeric(values, errors="coerce")
    finite = alp.dropna()
    if len(finite) and float(finite.max()) <= 1.5:
        print(
            "[tbportals] WARNING: ALP max <= 1.5 — assuming 0-1 encoding, rescaling x100. "
            "Verify this matches the radiologist scale (Kantipudi uses 0-100)."
        )
        alp = alp * 100.0
    return alp.clip(lower=0.0, upper=100.0)


def _coerce_cavity(values: pd.Series) -> pd.Series:
    """Coerce cavity to int {0, 1} from numeric or yes/no/bool tokens."""

    def _one(v: object) -> float:
        if pd.isna(v):
            return np.nan
        s = str(v).strip().lower()
        if s in _TRUE_TOKENS:
            return 1.0
        if s in _FALSE_TOKENS:
            return 0.0
        try:
            return 1.0 if float(s) > 0.5 else 0.0
        except ValueError:
            return np.nan

    return values.map(_one)


def _winlong(path: str) -> str:
    """Return an absolute path string with the ``\\?\\`` prefix on Windows.

    pathlib.Path does NOT reliably support the ``\\?\\`` prefix, so this helper
    works entirely with plain strings and os.path. The prefix tells Win32 to
    skip the 260-char MAX_PATH limit, which TB Portals DICOM trees routinely
    exceed.
    """
    abs_path = os.path.abspath(path)
    if os.name == "nt" and not abs_path.startswith("\\\\?\\"):
        return "\\\\?\\" + abs_path
    return abs_path


# ---------------------------------------------------------------------------
# Zip-encoded DICOM reference helpers
# ---------------------------------------------------------------------------

# Separator that cannot appear in any file path or DICOM UID.
_ZIP_SEP = "!DICOM!"

# Common top-level prefixes in TB Portals CXR zips that are NOT part of the
# series_instance_content_url path in the annotation CSV.
_ZIP_STRIP_PREFIXES = ("GlobalBucket/", "globalBucket/", "globalbucket/", "TB_Portals_CXRs/")


def _encode_zip_ref(zip_path: str, entry: str) -> str:
    """Encode a zip+entry pair as a single path string for the manifest CSV."""
    return zip_path + _ZIP_SEP + entry


def _is_zip_ref(p: str) -> bool:
    return isinstance(p, str) and _ZIP_SEP in p


def _decode_zip_ref(p: str) -> tuple[str, str]:
    """Split a zip-ref string into (zip_path, zip_entry_name)."""
    i = p.index(_ZIP_SEP)
    return p[:i], p[i + len(_ZIP_SEP):]


def _strip_bucket_prefix(entry_name: str) -> str:
    """Remove leading GlobalBucket (or similar) prefix from a zip entry name."""
    for prefix in _ZIP_STRIP_PREFIXES:
        if entry_name.startswith(prefix):
            return entry_name[len(prefix):]
    return entry_name


def _find_cxr_root_str(root_str: str) -> str:
    """Return the first child directory whose name contains 'CXR', or root_str.

    TB Portals ships images inside a subfolder like
    ``TB_Portals_Published_CXRs_August_2023`` inside the root drive folder.
    Uses os.scandir (string-based) so the ``\\?\\`` prefix works correctly.
    """
    try:
        for entry in os.scandir(_winlong(root_str)):
            if entry.is_dir() and "CXR" in entry.name.upper():
                return entry.path
    except (PermissionError, OSError):
        pass
    return root_str


def _find_entry_in_zip(z: zipfile.ZipFile, keyword: str) -> str | None:
    """Return the first zip entry whose name contains ``keyword``.

    Normalises both sides (lowercase, strip spaces and underscores) before
    matching, so 'TB Portals CXRs' finds 'TB_Portals_CXRs_August_2023.csv'.
    """
    def _norm(s: str) -> str:
        return s.lower().replace(" ", "").replace("_", "")

    kw = _norm(keyword)
    for name in z.namelist():
        if kw in _norm(name) and name.lower().endswith((".csv", ".xlsx")):
            return name
    return None


# ---------------------------------------------------------------------------
# Public: DICOM index
# ---------------------------------------------------------------------------

def build_dicom_index(
    image_roots: list[str | Path],
    *,
    verbose: bool = True,
) -> dict[str, str]:
    """Walk image_roots recursively; return {key: path_str} for all .dcm files.

    Each DICOM is indexed under multiple keys so look-ups work regardless of
    the format stored in ``series_instance_content_url``:
    - relative path as in zip (``uuid/study/series/instance.dcm``)
    - filename only (``instance.dcm``)
    - stem only    (``instance``)

    Supports two source types in the same call:
    - A directory path: scans for loose ``.dcm`` files AND ``.zip`` files that
      contain DICOMs (TB Portals part-zip format).
    - A ``.zip`` file path directly: scans entries inside that zip.

    Zip entries are returned as ``"zip_path!DICOM!entry_name"`` strings that
    ``src.data.tbportals_dataset._load_image`` understands.

    For best training throughput, extract the CXRs first with
    ``scripts/extract_cxr_dcm.ps1`` and then point image_roots at the output
    directory (``E:\\dcm``).  That reduces every image load to a plain file
    read instead of opening a zip on every batch.
    """
    index: dict[str, str] = {}

    def _index_zip(zip_path_str: str) -> int:
        count = 0
        try:
            with zipfile.ZipFile(zip_path_str, "r") as z:
                for info in z.infolist():
                    if info.is_dir() or not info.filename.lower().endswith(".dcm"):
                        continue
                    value = _encode_zip_ref(zip_path_str, info.filename)
                    # Strip common bucket prefix so series_instance_content_url matches.
                    rel = _strip_bucket_prefix(info.filename)
                    fname = rel.split("/")[-1]
                    stem = os.path.splitext(fname)[0]
                    for key in (rel, fname, stem, stem + ".dcm"):
                        if key:
                            index.setdefault(key, value)
                    count += 1
        except Exception as exc:
            if verbose:
                print(f"[tbportals]   failed to read zip {zip_path_str[:80]}: {exc}")
        return count

    for root in image_roots:
        root_str = str(root)
        count = 0

        # Accept a direct .zip file as a root (e.g. the Archive CXR zip).
        if root_str.lower().endswith(".zip"):
            count = _index_zip(root_str)
            if verbose:
                print(f"[tbportals] indexed {count} DICOMs in zip {root_str}")
            continue

        # Walk directory: index loose .dcm files AND .zip files containing DICOMs.
        walk_errors: list[str] = []
        for dirpath, _, filenames in os.walk(
            root_str, onerror=lambda e: walk_errors.append(str(e))
        ):
            for fn in filenames:
                lower_fn = fn.lower()
                full = os.path.join(dirpath, fn)

                if lower_fn.endswith(".dcm"):
                    stem = os.path.splitext(fn)[0]
                    index.setdefault(fn, full)
                    index.setdefault(stem, full)
                    count += 1

                elif lower_fn.endswith(".zip"):
                    zip_count = _index_zip(full)
                    count += zip_count

        if walk_errors and verbose:
            print(
                f"[tbportals]   walk errors in {root_str} "
                f"({len(walk_errors)} dirs): {walk_errors[0][:120]}"
            )
        if count == 0 and verbose:
            try:
                entries = [e.name for e in os.scandir(root_str)]
                print(
                    f"[tbportals]   0 DICOMs found under {root_str}. "
                    f"First-level entries: {entries[:12]}"
                )
            except Exception:
                pass
        if verbose:
            print(f"[tbportals] indexed {count} DICOMs under {root_str}")

    unique = len(set(index.values()))
    print(f"[tbportals] DICOM index total: {unique} unique DICOMs")
    return index


# ---------------------------------------------------------------------------
# Public: CSV extraction helper
# ---------------------------------------------------------------------------

def extract_csv_from_zip(
    zip_path: str | Path,
    out_dir: str | Path,
    *,
    csv_name: str | None = None,
) -> Path:
    """Extract the first (or named) CSV/XLSX from a zip and return its local path.

    Use this to inspect the annotation tables before building the manifest:

        csv_path = extract_csv_from_zip(ZIP_PATH, WORK / 'csv', csv_name='CXRs')
        pd.read_csv(csv_path).columns.tolist()
    """
    zip_path = Path(zip_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path) as z:
        all_csvs = [n for n in z.namelist() if n.lower().endswith((".csv", ".xlsx"))]
        if not all_csvs:
            raise FileNotFoundError(
                f"No CSV/XLSX inside {zip_path.name}. "
                f"Contents sample: {z.namelist()[:20]}"
            )
        target = _find_entry_in_zip(z, csv_name) if csv_name else all_csvs[0]
        if target is None or target not in z.namelist():
            raise FileNotFoundError(
                f"No CSV matching {csv_name!r} in zip. Available: {all_csvs}"
            )
        out_file = out_dir / Path(target).name
        with z.open(target) as src, open(out_file, "wb") as dst:
            dst.write(src.read())
    print(f"[tbportals] extracted -> {out_file}")
    return out_file


def list_csvs_in_zip(zip_path: str | Path) -> list[str]:
    """Return all CSV/XLSX names inside the zip (useful for inspection)."""
    with zipfile.ZipFile(zip_path) as z:
        return [n for n in z.namelist() if n.lower().endswith((".csv", ".xlsx"))]


# ---------------------------------------------------------------------------
# Public: Manifest builders
# ---------------------------------------------------------------------------


def _aggregate_annotations(ann_df: pd.DataFrame, join_key: str) -> pd.DataFrame:
    """Collapse sextant-level annotation rows to one row per image.

    The manual-annotations CSV has one row per sextant (up to 6 per image).
    Overall-lung-level columns (``overallpercentofabnormalvolume``,
    ``timika_score``, ``rater``) are the same value in every sextant row, so
    we take the first. Binary cavity is derived by checking whether any sextant
    has a positive count in ``smallcavities``, ``mediumcavities``, or
    ``largecavities``.
    """
    # Columns that are repeated at the overall-lung level — take the first value.
    overall_cols = [
        c for c in ["overallpercentofabnormalvolume", "timika_score", "rater"]
        if c in ann_df.columns
    ]
    per_image = ann_df.groupby(join_key)[overall_cols].first().reset_index()

    # Derive binary cavity: 1 if any sextant row has any cavity count > 0.
    cavity_src_cols = [
        c for c in ["smallcavities", "mediumcavities", "largecavities"]
        if c in ann_df.columns
    ]
    if cavity_src_cols:
        has_cavity = (
            ann_df.groupby(join_key)[cavity_src_cols]
            .apply(lambda df: bool((df.fillna(0) > 0).any(axis=None)))
            .reset_index(name="cavitiespresent")
        )
        has_cavity["cavitiespresent"] = has_cavity["cavitiespresent"].astype(int)
        per_image = per_image.merge(has_cavity, on=join_key, how="left")
        per_image["cavitiespresent"] = per_image["cavitiespresent"].fillna(0).astype(int)
    else:
        print("[tbportals] WARNING: no smallcavities/mediumcavities/largecavities columns found.")
        per_image["cavitiespresent"] = 0

    print(
        f"[tbportals] aggregated {len(ann_df)} sextant rows "
        f"-> {len(per_image)} image-level rows "
        f"(cavity rate {per_image['cavitiespresent'].mean():.2%})"
    )
    return per_image


def build_manifest_from_tbportals_zip(
    zip_path: str | Path,
    image_roots: str | Path | list[str | Path],
    out_path: str | Path,
    *,
    column_map: dict[str, str] | None = None,
    dicom_index: dict[str, Path] | None = None,
) -> pd.DataFrame:
    """Build the manifest directly from the TB Portals Published Imaging zip.

    Reads ``TB Portals CXRs.csv`` (image metadata) and
    ``TB Portals Published CXR Manual Annotations.csv`` (radiologist labels),
    joins them on ``imagingstudy_ID``, then calls :func:`build_manifest_from_raw`.

    Parameters
    ----------
    zip_path:
        Path to ``TB_Portals_Published_Imaging_Data_YYYYMMDD.zip``.
    image_roots:
        One or more directories to search for DICOM files (e.g.
        ``[r'E:\\Archive', r'E:\\March2025', r'E:\\March2026']``).
    out_path:
        Where to write the final manifest CSV.
    column_map:
        Override specific COLUMN_MAP entries (e.g. if cavity column is named
        differently in your release) — no source edits needed.
    dicom_index:
        Pre-built index from :func:`build_dicom_index`. Pass this when calling
        this function multiple times to avoid re-walking the disk.
    """
    cmap = {**COLUMN_MAP, **(column_map or {})}
    zip_path = Path(zip_path)

    with zipfile.ZipFile(zip_path) as z:
        cxr_entry = _find_entry_in_zip(z, TBPORTALS_CXR_CSV)
        ann_entry = _find_entry_in_zip(z, TBPORTALS_ANNOTATIONS_CSV)

        all_csvs = [n for n in z.namelist() if n.lower().endswith((".csv", ".xlsx"))]
        if cxr_entry is None or ann_entry is None:
            raise FileNotFoundError(
                f"Could not locate both required CSVs in {zip_path.name}.\n"
                f"  Looking for: '{TBPORTALS_CXR_CSV}' and '{TBPORTALS_ANNOTATIONS_CSV}'\n"
                f"  Available CSVs: {all_csvs}\n"
                f"  If your zip uses different names, pass the exact names via "
                f"  column_map={{'...': '...'}} or extract manually with extract_csv_from_zip()."
            )

        print(f"[tbportals] CXR metadata CSV:  {cxr_entry}")
        print(f"[tbportals] Annotations CSV:   {ann_entry}")

        with z.open(cxr_entry) as f:
            cxr_df = pd.read_csv(f, low_memory=False)
        with z.open(ann_entry) as f:
            ann_df = pd.read_csv(f, low_memory=False)

    print(f"[tbportals] CXR metadata cols:  {list(cxr_df.columns)}")
    print(f"[tbportals] Annotations cols:   {list(ann_df.columns)}")

    if JOIN_KEY not in cxr_df.columns or JOIN_KEY not in ann_df.columns:
        raise KeyError(
            f"Join key '{JOIN_KEY}' missing from one or both tables.\n"
            f"  CXR cols: {list(cxr_df.columns)}\n"
            f"  Ann cols: {list(ann_df.columns)}"
        )

    # Keep only the columns we need to avoid suffix collisions after merge.
    alp_col    = cmap["alp"]
    cavity_col = cmap["cavity"]
    img_col    = cmap["image_id"]
    pat_col    = cmap["patient_id"]
    ctry_col   = cmap["country"]

    def _keep(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
        present = [c for c in cols if c in df.columns]
        missing = [c for c in cols if c not in df.columns]
        if missing:
            print(f"[tbportals] NOTE: columns not found (will error later): {missing}")
            print(f"[tbportals]       available in that table: {list(df.columns)}")
        return df[present].copy()

    cxr_sub = _keep(cxr_df, [JOIN_KEY, img_col, pat_col, ctry_col])

    # Aggregate sextant-level annotations to one row per image before joining.
    # This derives cavitiespresent from per-sextant cavity counts.
    ann_agg = _aggregate_annotations(ann_df, JOIN_KEY)
    ann_sub = _keep(ann_agg, [JOIN_KEY, alp_col, "cavitiespresent"])

    merged = cxr_sub.merge(ann_sub, on=JOIN_KEY, how="inner")
    print(f"[tbportals] joined table: {len(merged)} rows with both image metadata and annotations")

    # Write merged table to a temp CSV and delegate to build_manifest_from_raw.
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".csv")
    try:
        os.close(tmp_fd)
        merged.to_csv(tmp_path, index=False)
        df = build_manifest_from_raw(
            tmp_path, image_roots, out_path,
            column_map=cmap, dicom_index=dicom_index,
        )
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    return df


def build_manifest_from_raw(
    raw_csv_path: str | Path,
    image_roots: str | Path | list[str | Path],
    out_path: str | Path,
    *,
    column_map: dict[str, str] | None = None,
    dicom_index: dict[str, Path] | None = None,
) -> pd.DataFrame:
    """Read a pre-joined annotation CSV and write the manifest.

    ``image_roots`` may be a single directory or a list of directories.
    Pass a pre-built ``dicom_index`` (from :func:`build_dicom_index`) to avoid
    re-walking the disk — important when calling this function multiple times.

    Image path resolution order:
    1. DICOM index lookup (filename, stem, stem+.dcm)
    2. Relative path join: ``image_root / series_instance_content_url``
    3. Relative path join from the CXR subfolder inside each root
    """
    cmap = {**COLUMN_MAP, **(column_map or {})}
    raw_csv_path = Path(raw_csv_path)

    if isinstance(image_roots, (str, Path)):
        roots: list[str] = [str(image_roots)]
    else:
        roots = [str(r) for r in image_roots]

    if raw_csv_path.suffix.lower() in {".xlsx", ".xls"}:
        raw = pd.read_excel(raw_csv_path)
    else:
        raw = pd.read_csv(raw_csv_path)

    missing_cols = [src for src in cmap.values() if src not in raw.columns]
    if missing_cols:
        raise KeyError(
            f"Columns {missing_cols} not found in {raw_csv_path.name}.\n"
            f"Available: {list(raw.columns)}\n"
            f"Fix COLUMN_MAP or pass column_map={{...}} to override."
        )

    df = pd.DataFrame(
        {
            "image_id":   raw[cmap["image_id"]].astype(str),
            "image_path": raw[cmap["image_path"]].astype(str),
            "patient_id": raw[cmap["patient_id"]].astype(str),
            "country":    raw[cmap["country"]].astype(str).str.strip(),
            "alp_0_100":  _coerce_alp(raw[cmap["alp"]]),
            "cavity":     _coerce_cavity(raw[cmap["cavity"]]),
        }
    )

    idx = dicom_index or {}

    def _resolve(raw_val: str) -> str | None:
        fname = os.path.basename(raw_val)
        stem = os.path.splitext(fname)[0]

        # 1. DICOM index lookup (covers both "file.dcm" and plain "file" keys).
        for key in (raw_val, fname, stem, stem + ".dcm"):
            if key in idx:
                return idx[key]

        # 2. Direct relative-path join from each root and its CXR subfolder.
        rel = raw_val.replace("/", os.sep).lstrip(os.sep)
        for root in roots:
            for base in (root, _find_cxr_root_str(root)):
                try:
                    candidate = _winlong(os.path.join(base, rel))
                    if os.path.isfile(candidate):
                        return candidate
                except (OSError, ValueError):
                    pass

        return None

    df["image_path"] = df["image_path"].map(_resolve)

    n0 = len(df)
    df = df.dropna(subset=["alp_0_100", "cavity"])
    df["cavity"] = df["cavity"].astype(int)

    def _file_exists(p: object) -> bool:
        if not p or not isinstance(p, str):
            return False
        if _is_zip_ref(p):
            return True  # entry was verified while building the index
        try:
            return os.path.isfile(_winlong(p))
        except (OSError, ValueError):
            return False

    exists = df["image_path"].map(_file_exists)
    n_missing_img = int((~exists).sum())
    df = df[exists].reset_index(drop=True)

    print(
        f"[tbportals] kept {len(df)}/{n0} rows "
        f"(dropped {n0 - len(df)}: missing labels or images; {n_missing_img} missing image files)."
    )
    if n_missing_img > 0 and len(df) == 0:
        print(
            "[tbportals] WARNING: ALL rows dropped due to missing images.\n"
            "  Check that image_roots point to the correct directories.\n"
            "  Run build_dicom_index(image_roots) and inspect the returned dict.\n"
            "  The series_instance_content_url values in the CSV must match "
            "the DICOM filenames or their stems."
        )

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    summarize_manifest(df)
    return df


# ---------------------------------------------------------------------------
# Public: Split + leakage guard
# ---------------------------------------------------------------------------

def load_manifest(path: str | Path) -> pd.DataFrame:
    """Load and validate a manifest CSV."""
    df = pd.read_csv(path, dtype={"image_id": str, "patient_id": str, "country": str})
    missing = [c for c in MANIFEST_COLUMNS if c not in df.columns]
    if missing:
        raise KeyError(f"Manifest {path} missing columns {missing}. Expected {MANIFEST_COLUMNS}.")
    df["alp_0_100"] = df["alp_0_100"].astype(float)
    df["cavity"] = df["cavity"].astype(int)
    df["country"] = df["country"].str.strip()
    return df


def assert_no_patient_leakage(*frames: pd.DataFrame) -> None:
    """Hard assertion: no ``patient_id`` is shared across the given splits.

    Patient leakage is the #1 silent metric inflator on TB Portals (multiple
    CXRs per patient). This must never fail.
    """
    sets = [set(f["patient_id"]) for f in frames]
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            overlap = sets[i] & sets[j]
            if overlap:
                raise AssertionError(
                    f"Patient leakage between split {i} and split {j}: "
                    f"{len(overlap)} shared patient_id(s), e.g. {sorted(overlap)[:5]}."
                )


def make_country_split(
    df: pd.DataFrame,
    held_out_country: str,
    *,
    val_fraction: float = 0.2,
    seed: int = 1337,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Country-segregated, patient-disjoint split matching Kantipudi.

    - test  = every image of ``held_out_country``
    - train/val = remaining countries, split by PATIENT (~1-val_fraction / val_fraction)
    """
    if held_out_country not in set(df["country"]):
        raise ValueError(
            f"Held-out country {held_out_country!r} not present. "
            f"Available: {sorted(df['country'].unique())}."
        )

    test_df = df[df["country"] == held_out_country].reset_index(drop=True)
    pool = df[df["country"] != held_out_country].reset_index(drop=True)

    patients = np.asarray(pool["patient_id"].unique(), dtype=object)
    rng = np.random.default_rng(seed)
    rng.shuffle(patients)
    n_val = max(1, int(round(len(patients) * val_fraction)))
    val_patients = set(patients[:n_val])

    val_df   = pool[pool["patient_id"].isin(val_patients)].reset_index(drop=True)
    train_df = pool[~pool["patient_id"].isin(val_patients)].reset_index(drop=True)

    assert_no_patient_leakage(train_df, val_df, test_df)
    print(
        f"[tbportals] split held_out={held_out_country}: "
        f"train={len(train_df)} val={len(val_df)} test={len(test_df)} "
        f"(train/val patients {len(set(train_df['patient_id']))}/{len(val_patients)})."
    )
    return train_df, val_df, test_df


def summarize_manifest(df: pd.DataFrame) -> pd.DataFrame:
    """Print and return per-country counts, ALP mean/std, and cavity rate."""
    g = (
        df.groupby("country")
        .agg(
            n_images=("image_id", "size"),
            n_patients=("patient_id", "nunique"),
            alp_mean=("alp_0_100", "mean"),
            alp_std=("alp_0_100", "std"),
            cavity_rate=("cavity", "mean"),
        )
        .round(2)
    )
    print("[tbportals] manifest summary:")
    print(g.to_string())
    print(f"[tbportals] TOTAL images={len(df)} patients={df['patient_id'].nunique()}")
    return g


def make_synthetic_dataset(
    out_dir: str | Path,
    *,
    n: int = 600,
    countries: tuple[str, ...] = (
        "Georgia", "Belarus", "Ukraine", "Romania", "Moldova", "Kazakhstan"
    ),
    img_size: int = 256,
    seed: int = 1337,
) -> Path:
    """Write tiny random grayscale PNGs + a manifest to exercise the full stack
    before the TB Portals data arrives. Returns the manifest path."""
    from PIL import Image

    out_dir = Path(out_dir)
    img_dir = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    rows = []
    n_patients = max(2, n // 3)
    for i in range(n):
        country = countries[i % len(countries)]
        patient_id = f"{country[:2].upper()}-{rng.integers(0, n_patients)}"
        alp = float(rng.uniform(0, 100))
        cavity = int(rng.uniform(0, 1) < (0.15 + 0.5 * alp / 100.0))
        arr = rng.integers(0, 256, size=(img_size, img_size), dtype=np.uint8)
        fname = f"img_{i:05d}.png"
        Image.fromarray(arr, mode="L").save(img_dir / fname)
        rows.append(
            {
                "image_id":   f"img_{i:05d}",
                "image_path": str((img_dir / fname).resolve()),
                "patient_id": patient_id,
                "country":    country,
                "alp_0_100":  round(alp, 2),
                "cavity":     cavity,
            }
        )

    df = pd.DataFrame(rows, columns=MANIFEST_COLUMNS)
    manifest_path = out_dir / "manifest.csv"
    df.to_csv(manifest_path, index=False)
    print(f"[tbportals] synthetic dataset written: {manifest_path}")
    summarize_manifest(df)
    return manifest_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="TB Portals manifest builder.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    z = sub.add_parser("zip", help="Build manifest from the TB Portals Published Imaging zip.")
    z.add_argument("--zip",          required=True,  help="Path to TB_Portals_Published_Imaging_Data_*.zip")
    z.add_argument("--image-roots",  nargs="+",  required=True, help="Directories with DICOM files (space-separated).")
    z.add_argument("--out",          required=True,  help="Output manifest.csv path.")
    z.add_argument("--build-index",  action="store_true", help="Pre-build DICOM index (recommended for first run).")

    b = sub.add_parser("build", help="Build manifest from a pre-joined annotation CSV.")
    b.add_argument("--raw",          required=True,  help="Pre-joined annotation CSV/XLSX.")
    b.add_argument("--image-roots",  nargs="+", required=True, help="Directories with images (space-separated).")
    b.add_argument("--out",          required=True,  help="Output manifest.csv path.")

    s = sub.add_parser("synth", help="Write a synthetic dataset for stack testing.")
    s.add_argument("--out",  required=True, help="Output directory.")
    s.add_argument("--n",    type=int, default=600)
    s.add_argument("--seed", type=int, default=1337)

    v = sub.add_parser("summary", help="Summarize an existing manifest.")
    v.add_argument("--manifest", required=True)

    args = parser.parse_args(argv)

    if args.cmd == "zip":
        idx = build_dicom_index(args.image_roots) if args.build_index else None
        build_manifest_from_tbportals_zip(args.zip, args.image_roots, args.out, dicom_index=idx)
    elif args.cmd == "build":
        build_manifest_from_raw(args.raw, args.image_roots, args.out)
    elif args.cmd == "synth":
        make_synthetic_dataset(args.out, n=args.n, seed=args.seed)
    elif args.cmd == "summary":
        summarize_manifest(load_manifest(args.manifest))


if __name__ == "__main__":
    _main()
