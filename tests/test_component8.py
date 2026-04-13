import numpy as np
import torch
from src.components.component8_metrics import compute_timika_score

def test_compute_timika_score_mild_no_cavity():
    """Test scenario with low ALP and no cavities, resulting in mild severity."""
    # Mock lung mask (800x800 = 640,000 pixels)
    lung_mask = np.zeros((1, 1024, 1024), dtype=np.float32)
    lung_mask[0, 100:900, 100:900] = 1.0 
    
    # Mock lesion mask (~64,000 pixels, exactly 10% of lung)
    mask_refined = np.zeros((1, 1024, 1024), dtype=np.float32)
    mask_refined[0, 100:300, 100:420] = 1.0 
    
    # Mock cavity mask (No cavity)
    mask_e2 = np.zeros((1, 256, 256), dtype=np.float32) 
    
    metrics = compute_timika_score(mask_refined, lung_mask, mask_e2)
    
    assert np.isclose(metrics["ALP"], 10.0)
    assert metrics["cavity_flag"] == 0
    assert np.isclose(metrics["timika_score"], 10.0)
    assert metrics["severity"] == 'mild'

def test_compute_timika_score_severe_with_cavity():
    """Test scenario with >50% ALP and a valid cavity, tracking proper Timika penalty."""
    lung_mask = np.zeros((1024, 1024), dtype=np.float32)
    lung_mask[0:1000, 0:1000] = 1.0 # 1,000,000 pixels
    
    mask_refined = np.zeros((1024, 1024), dtype=np.float32)
    mask_refined[0:600, 0:1000] = 1.0 # 60% of lung
    
    mask_e2 = np.zeros((256, 256), dtype=np.float32)
    mask_e2[50:60, 50:60] = 1.0 # 10x10 = 100 pixels (>50 threshold -> flag = 1)
    
    metrics = compute_timika_score(mask_refined, lung_mask, mask_e2)
    
    assert np.isclose(metrics["ALP"], 60.0)
    assert metrics["cavity_flag"] == 1
    assert np.isclose(metrics["timika_score"], 100.0) # 60 + 40
    assert metrics["severity"] == 'severe'

def test_cavity_size_filtering():
    """Ensure small potential cavities are naturally ignored (e.g. noise points < 50 pixels)."""
    lung_mask = np.ones((1024, 1024), dtype=np.float32)
    mask_refined = np.zeros((1024, 1024), dtype=np.float32)
    
    # 49 pixels (should be filtered out since s > 50 is required)
    mask_e2 = np.zeros((256, 256), dtype=np.float32)
    mask_e2[10:17, 10:17] = 1.0 # 7x7 = 49
    
    metrics1 = compute_timika_score(mask_refined, lung_mask, mask_e2)
    assert metrics1["cavity_flag"] == 0
    
    # 51 pixels (should trigger flag)
    mask_e2[20:30, 20:26] = 1.0 # 10x6 = 60 pixels
    metrics2 = compute_timika_score(mask_refined, lung_mask, mask_e2)
    assert metrics2["cavity_flag"] == 1

def test_torch_tensor_acceptance():
    """Verify PyTorch tensors are safely converted to NumPy inside the function."""
    lung_mask = torch.ones((1, 1024, 1024))
    mask_refined = torch.ones((1, 1024, 1024)) * 0.6
    mask_e2 = torch.zeros((1, 256, 256))
    
    metrics = compute_timika_score(mask_refined, lung_mask, mask_e2)
    assert np.isclose(metrics["ALP"], 100.0)
    assert metrics["timika_score"] == 100.0
    assert metrics["severity"] == 'severe'
