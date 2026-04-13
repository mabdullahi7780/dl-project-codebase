import torch
import torch.nn as nn
from src.components.component7_verification import Component7aBoundaryCritic, Component7bFPAuditor

def train_boundary_critic():
    model = Component7aBoundaryCritic()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
    criterion = nn.BCELoss()
    
    # Mock data
    x = torch.randn(8, 3, 224, 224)
    y = torch.rand(8, 1)
    
    model.train()
    optimizer.zero_grad()
    bd_score = model(x)
    loss = criterion(bd_score, y)
    loss.backward()
    optimizer.step()
    
    print(f"Boundary Critic Loss: {loss.item()}")

def train_fp_auditor():
    model = Component7bFPAuditor(txh_fallback=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    # FP class weighted 3x
    # Since it's binary, we use BCELoss with a weight for positive class
    # For BCEWithLogits, we would use pos_weight. For standard BCELoss:
    def weighted_bce_loss(pred, target, fp_weight=3.0):
        weight = torch.where(target == 1, fp_weight, 1.0)
        return nn.functional.binary_cross_entropy(pred, target, weight=weight)
        
    x = torch.randn(8, 1, 224, 224)
    y = torch.round(torch.rand(8, 1)) # Binary targets
    
    model.train()
    optimizer.zero_grad()
    fp_prob, _ = model(x)
    loss = weighted_bce_loss(fp_prob, y)
    loss.backward()
    optimizer.step()
    
    print(f"FP Auditor Loss: {loss.item()}")

if __name__ == '__main__':
    train_boundary_critic()
    train_fp_auditor()
