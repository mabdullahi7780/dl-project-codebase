from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from src.app.infer import run_single_image_inference


def test_single_image_inference_writes_expected_artifacts(tmp_path: Path) -> None:
    image = np.linspace(0, 255, 700 * 700, dtype=np.uint8).reshape(700, 700)
    image_path = tmp_path / "demo.png"
    Image.fromarray(image, mode="L").save(image_path)

    bundle = run_single_image_inference(
        image_path=image_path,
        dataset="nih",
        outdir=tmp_path / "outputs",
        view="PA",
    )

    assert bundle.evidence_json is not None
    assert bundle.report_text is not None
    assert (tmp_path / "outputs" / "evidence.json").exists()
    assert (tmp_path / "outputs" / "report.txt").exists()
    assert (tmp_path / "outputs" / "overlay.png").exists()
