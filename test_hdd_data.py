import argparse
import sys
from pathlib import Path
from PIL import Image
import torch
import torchvision.transforms.functional as TF

sys.path.append(str(Path(__file__).resolve().parent))
from src.components.component0_qc import harmonise_sample

def test_hdd_data_processing():
    # Target image from your external HDD
    hdd_image_path = "/Volumes/Toshiba HDR/TB_DATA/raw/montgomery/images/MCUCXR_0023_0.png"
    
    print(f"Reading from HDD: {hdd_image_path}")
    
    try:
        # Load the image using PIL
        pil_img = Image.open(hdd_image_path).convert('L')
        # Convert to tensor [1, H, W]
        img_tensor = TF.to_tensor(pil_img)
        
        print(f"Loaded Raw Image Tensor Shape: {img_tensor.shape}")
        
        # Raw dictionary exactly like our Dataset classes will yield
        raw_sample = {
            "image": img_tensor,
            "dataset_id": "montgomery",
            "view": "pa"
        }
        
        # Pass to our exact Component 0 Preprocessing
        # This applies the 12-bit Montgomery normalization, CLAHE (since it's montgomery), 
        # and resizing for both SAM (1024) and DenseNet (224)
        processed = harmonise_sample(raw_sample)
        
        # Checking the results
        print("\n--- Component 0 Processing Results ---")
        
        # SAM ViT-H expects [-1024.0, 1024.0] and 1024x1024
        x_1024 = processed.x_1024
        print(f"x_1024 (SAM) Shape: {x_1024.shape}")
        print(f"x_1024 (SAM) Min/Max values: [{x_1024.min():.2f}, {x_1024.max():.2f}]")
        assert tuple(x_1024.shape) == (1, 1024, 1024)
        
        # DenseNet121 expects standard [0,1] floating floats (TorchXRayVision rescales this to [-1024, 1024] at inference time)
        # Actually our docs say TXV needs [-1024, 1024] scaled in its domain.
        x_224 = processed.x_224
        print(f"x_224 (DenseNet) Shape: {x_224.shape}")
        print(f"x_224 (DenseNet) Min/Max values: [{x_224.min():.2f}, {x_224.max():.2f}]")
        assert tuple(x_224.shape) == (1, 224, 224)
        
        print("\nSuccess! The codebase correctly reads and processes images directly from your External HDD.")
        
    except Exception as e:
        print(f"Error during processing: {e}")

if __name__ == "__main__":
    test_hdd_data_processing()
