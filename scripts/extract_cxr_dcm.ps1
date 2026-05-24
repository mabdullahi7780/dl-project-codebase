# extract_cxr_dcm.ps1
# Extracts CXR DICOMs from all TB Portals zip archives into E:\dcm\
# Ignores CT folders.  Strips the "GlobalBucket/" prefix from the Archive zip
# so all images land at:  E:\dcm\<patient-uuid>\<study>\<series>\<instance>.dcm
#
# Run from any directory in PowerShell (no activation needed):
#   powershell -ExecutionPolicy Bypass -File scripts\extract_cxr_dcm.ps1
#
# Expected sources:
#   E:\Archive\TB_Portals_Published_CXRs_August_2023_Amended.zip   (1 zip, ~41k DICOMs)
#   E:\March2025\TB_Portals_CXRs_March_2025\*.zip                  (18 part-zips)
#   E:\March2026\TB_Portals_CXRs_March_2026\*.zip                  (20 part-zips)
#
# Output:  E:\dcm\   (all DICOMs, flat uuid/study/series/instance structure)

$ErrorActionPreference = "Stop"
Add-Type -AssemblyName System.IO.Compression.FileSystem

$DEST = "E:\dcm"

# ── Helper: extract a single zip, stripping an optional prefix ────────────────
function Extract-ZipCXR {
    param(
        [string]$ZipPath,
        [string]$StripPrefix = ""
    )

    Write-Host "  Extracting: $(Split-Path $ZipPath -Leaf)"
    $archive = [System.IO.Compression.ZipFile]::OpenRead($ZipPath)
    $count   = 0
    $errors  = 0

    foreach ($entry in $archive.Entries) {
        # Skip directory entries
        if ($entry.Name -eq "") { continue }
        # CXR DICOMs only
        if (-not $entry.FullName.ToLower().EndsWith(".dcm")) { continue }

        $relPath = $entry.FullName
        if ($StripPrefix -and $relPath.StartsWith($StripPrefix)) {
            $relPath = $relPath.Substring($StripPrefix.Length)
        }

        # Replace forward slashes with backslashes
        $relPath  = $relPath.Replace("/", "\")
        $destFile = [System.IO.Path]::Combine($DEST, $relPath)
        $destDir  = [System.IO.Path]::GetDirectoryName($destFile)

        if (-not [System.IO.Directory]::Exists($destDir)) {
            [System.IO.Directory]::CreateDirectory($destDir) | Out-Null
        }

        # Skip if already extracted
        if ([System.IO.File]::Exists($destFile)) {
            $count++
            continue
        }

        try {
            $srcStream  = $entry.Open()
            $dstStream  = [System.IO.File]::Create($destFile)
            $srcStream.CopyTo($dstStream)
            $dstStream.Close()
            $srcStream.Close()
            $count++
        } catch {
            $errors++
            if ($errors -le 5) {
                Write-Warning "    Failed to extract $relPath : $_"
            }
        }
    }

    $archive.Dispose()
    Write-Host "    -> $count DICOMs extracted ($errors errors)"
    return $count
}

# ── Create output directory ───────────────────────────────────────────────────
if (-not (Test-Path $DEST)) {
    New-Item -ItemType Directory -Path $DEST | Out-Null
}
Write-Host "Output directory: $DEST"
Write-Host ""

$total = 0

# ── 1. August 2023 Archive zip (has GlobalBucket/ prefix) ───────────────────
$archiveZip = "E:\Archive\TB_Portals_Published_CXRs_August_2023_Amended.zip"
if (Test-Path $archiveZip) {
    Write-Host "=== Archive (August 2023) ==="
    $total += Extract-ZipCXR -ZipPath $archiveZip -StripPrefix "GlobalBucket/"
} else {
    Write-Warning "Not found: $archiveZip"
}

Write-Host ""

# ── 2. March 2025 part-zips ──────────────────────────────────────────────────
$march25Dir = "E:\March2025\TB_Portals_CXRs_March_2025"
if (Test-Path $march25Dir) {
    Write-Host "=== March 2025 ==="
    $zips25 = Get-ChildItem $march25Dir -Filter "*.zip" | Sort-Object Name
    Write-Host "  Found $($zips25.Count) part-zips"
    foreach ($z in $zips25) {
        $total += Extract-ZipCXR -ZipPath $z.FullName
    }
} else {
    Write-Warning "Not found: $march25Dir"
}

Write-Host ""

# ── 3. March 2026 part-zips ──────────────────────────────────────────────────
$march26Dir = "E:\March2026\TB_Portals_CXRs_March_2026"
if (Test-Path $march26Dir) {
    Write-Host "=== March 2026 ==="
    $zips26 = Get-ChildItem $march26Dir -Filter "*.zip" | Sort-Object Name
    Write-Host "  Found $($zips26.Count) part-zips"
    foreach ($z in $zips26) {
        $total += Extract-ZipCXR -ZipPath $z.FullName
    }
} else {
    Write-Warning "Not found: $march26Dir"
}

Write-Host ""
Write-Host "=== Done: $total DICOMs total in $DEST ==="
Write-Host ""
Write-Host "Next step: run notebooks\local_01_build_manifest.ipynb"
Write-Host "  IMAGE_ROOTS is already set to E:\dcm"
