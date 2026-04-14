import torch
import torch.nn as nn
import torchvision.models as models

# 7a. Boundary Critic (ResNet18)
class Component7aBoundaryCritic(nn.Module):
    def __init__(self):
        super().__init__()
        # Initialize ResNet18
        # Using weights='DEFAULT' instead of deprecated pretrained=True
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        
        # Keep features up to the adaptive pool
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        
        # Freeze blocks 1-3
        # In ResNet18: 
        # layer1 is block 1, layer2 is block 2, layer3 is block 3
        # conv1, bn1, relu, maxpool are before layer1
        frozen_modules = [
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
            resnet.layer1, resnet.layer2, resnet.layer3
        ]
        
        for module in frozen_modules:
            for param in module.parameters():
                param.requires_grad = False
                
        # layer4 (block 4) and FC head are trainable
        for param in resnet.layer4.parameters():
            param.requires_grad = True
            
        # Recreate the classifier head
        # ResNet18 feature out is 512
        self.fc = nn.Sequential(
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x is [B, 3, 224, 224] - crop of CXR around predicted lesion centroid
        features = self.features(x) # [B, 512, 1, 1]
        features = features.view(features.size(0), -1) # [B, 512]
        bd_score = self.fc(features) # [B, 1] in [0, 1]
        return bd_score

# 7b. FP Auditor (DenseNet121)
class Component7bFPAuditor(nn.Module):
    def __init__(self, txh_fallback=True):
        super().__init__()
        
        # Load DenseNet121 backbone from TorchXRayVision
        try:
            import torchxrayvision as xrv
            self.backbone = xrv.models.DenseNet(weights="densenet121-res224-all")
            # We need both the penultimate features (1024) and the logits (18, if standard weights are 18 categories)
            # Actually, xrv DenseNet returns 18 logits. The penultimate features are before the classifier.
            # To extract penultimate features cleanly, we can use xrv's feature extraction:
            self.backbone.op_threshs = None # Disable operating thresholds output modification
        except ImportError:
            if txh_fallback:
                # Mock backbone for testing
                self.backbone = models.densenet121(weights=None)
                # Modify output to match txv 18 classes
                self.backbone.classifier = nn.Linear(1024, 18)
            else:
                raise ImportError("torchxrayvision not installed.")

        # Freeze ALL DenseNet121 backbone
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        # FP_head MLP: input 1042 (1024 + 18)
        self.fp_head = nn.Sequential(
            nn.Linear(1042, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # x is _224 [B, 1, 224, 224]
        # REUSE cached from Component 2 if possible. If passing through backbone:
        if isinstance(self.backbone, models.DenseNet):
            # Mock behavior
            x = x.repeat(1, 3, 1, 1) # densenet needs 3 channels if mock
            features = self.backbone.features(x)
            features = nn.functional.adaptive_avg_pool2d(features, (1, 1)).view(features.size(0), -1) # [B, 1024]
            logits = self.backbone.classifier(features) # [B, 18]
        else:
            # TXV behavior
            # TXV expects [B, 1, 224, 224]
            # It has a features method
            features = self.backbone.features(x) # [B, 1024, 7, 7]
            # global average pooling
            features = nn.functional.adaptive_avg_pool2d(features, (1, 1)).view(features.size(0), -1) # [B, 1024]
            logits = self.backbone(x) # [B, 18]
            
        # Concat [B, 1024] and [B, 18] -> [B, 1042]
        concat_feats = torch.cat([features, logits], dim=1)
        
        # FP_head
        fp_prob = self.fp_head(concat_feats)
        return fp_prob, concat_feats

# 7c. Refiner (MedSAM ViT-H re-prompt)
class Component7cRefiner(nn.Module):
    def __init__(self, expert3_decoder=None, sam_prompt_encoder=None):
        super().__init__()
        # Reuses MedSAM ViT-H, zero new weights
        self.sam_prompt_encoder = sam_prompt_encoder
        self.expert3_decoder = expert3_decoder
        
    def forward(self, image_embeddings, mask_fused, bd_score, mask_var, original_image):
        """
        IF bd_score < 0.7:
        1. uncertain_pts = (mask_var > 0.3).nonzero()
        2. Sample 5 point prompts from uncertain_pts along boundary
        3. Re-encode via SAM prompt encoder -> [5, 256]
        4. Run Expert 3 (boundary decoder) with new prompts -> mask_refined [B, 1, 256, 256]
        5. Arbiter: accept mask_refined if Dice improvement > 0 else keep mask_fused
        ELSE:
        mask_refined = mask_fused
        """
        B = image_embeddings.shape[0]
        mask_refined_list = []
        
        for i in range(B):
            if bd_score[i] < 0.7:
                # Need refinement
                uncertain_pts = (mask_var[i] > 0.3).nonzero()
                
                # Sample 5 points (mocking the selection logic here)
                if len(uncertain_pts) >= 5:
                    indices = torch.randperm(len(uncertain_pts))[:5]
                    prompts = uncertain_pts[indices]
                else:
                    prompts = uncertain_pts
                
                # In real MedSAM, we pass these point prompts to the prompt encoder
                if self.sam_prompt_encoder is not None and self.expert3_decoder is not None:
                    # Mocking the prompt encoding and expert3 decoding
                    # We would get sparse_embeddings from self.sam_prompt_encoder
                    # Then run self.expert3_decoder(image_embeddings[i:i+1], sparse_embeddings)
                    # For this implementation stub, we just return mask_fused or modified mask
                    pass
                
                # Arbiter logic would go here. For now, returning mask_fused as placeholder.
                mask_refined_list.append(mask_fused[i:i+1])
            else:
                # No refinement
                mask_refined_list.append(mask_fused[i:i+1])
                
        return torch.cat(mask_refined_list, dim=0)
