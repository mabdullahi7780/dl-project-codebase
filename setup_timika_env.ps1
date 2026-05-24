# setup_timika_env.ps1
# Creates the "timika" virtual environment and installs all dependencies.
#
# Run from the repo root in PowerShell:
#   powershell -ExecutionPolicy Bypass -File setup_timika_env.ps1
#
# After this script completes:
#   1. .\timika\Scripts\Activate.ps1
#   2. jupyter notebook
#   3. Open a local_*.ipynb and select kernel "Python (timika)"

$ErrorActionPreference = "Stop"

# ── Step 1: Create venv ───────────────────────────────────────────────────────
Write-Host "`n[1/5] Creating virtual environment 'timika'..." -ForegroundColor Cyan
python -m venv timika
if (-not $?) { Write-Error "python -m venv failed. Make sure Python 3.10+ is on PATH."; exit 1 }

# ── Step 2: Activate ──────────────────────────────────────────────────────────
Write-Host "[2/5] Activating..." -ForegroundColor Cyan
& .\timika\Scripts\Activate.ps1

# ── Step 3: Install PyTorch with CUDA ─────────────────────────────────────────
Write-Host "[3/5] Installing PyTorch with CUDA 12.1 (RTX 2070 compatible)..." -ForegroundColor Cyan
Write-Host "      If you need CUDA 11.8 instead, change 'cu121' to 'cu118' below."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Verify GPU is visible
python -c "import torch; print('CUDA available:', torch.cuda.is_available(), '| Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"

# ── Step 4: Install remaining requirements ────────────────────────────────────
Write-Host "[4/5] Installing remaining requirements..." -ForegroundColor Cyan
# Install without torch/torchvision (already installed with CUDA above)
pip install numpy pillow pyyaml pydantic scikit-learn pandas pytest matplotlib scipy jupyter ipykernel pydicom "pylibjpeg[all]" torchxrayvision opencv-python-headless

# ── Step 5: Register Jupyter kernel ───────────────────────────────────────────
Write-Host "[5/5] Registering Jupyter kernel as 'Python (timika)'..." -ForegroundColor Cyan
python -m ipykernel install --user --name timika --display-name "Python (timika)"

Write-Host "`n=== Setup complete ===" -ForegroundColor Green
Write-Host "To use:"
Write-Host "  1. Activate venv:   .\timika\Scripts\Activate.ps1"
Write-Host "  2. Start Jupyter:   jupyter notebook"
Write-Host "  3. Open:            notebooks\local_01_build_manifest.ipynb"
Write-Host "  4. Select kernel:   Python (timika)"
Write-Host ""
Write-Host "Run notebooks in order: local_01 -> local_03 -> local_04"
