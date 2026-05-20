import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ViT_B_16_Weights

class CrossScaleAttention(nn.Module):
    """
    Computes multi-head cross-attention between two scales (e.g. 20x and 40x).
    Permits one scale to query the other scale's representations.
    """
    def __init__(self, embed_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x_query, x_kv, return_attn=False):
        # x_query: [batch_size, 1, embed_dim]
        # x_kv: [batch_size, 1, embed_dim]
        attn_out, attn_weights = self.mha(x_query, x_kv, x_kv)
        out = self.norm(x_query + self.dropout(attn_out))
        if return_attn:
            return out, attn_weights
        return out

class MOPFN(nn.Module):
    """
    Multi-Scale Ordinal Pathology Foundation Network (MOPFN).
    Processes paired 20x and 40x pathology images using a shared ViT backbone,
    fuses them via bidirectional cross-attention, and classifies them across
    three clinical multi-task heads (Malignancy, Subtype, Ordinal Differentiation).
    """
    def __init__(self, embed_dim=768, num_heads=4, num_diff_classes=3, pretrained=True):
        super().__init__()
        
        # 1. Shared Vision Transformer Backbone
        # vit_b_16 expects input size 224x224 and yields 768-dim embeddings
        if pretrained:
            weights = ViT_B_16_Weights.DEFAULT
            self.backbone = models.vit_b_16(weights=weights)
        else:
            self.backbone = models.vit_b_16()
            
        # Extract features (exclude the default classification head of the ViT model)
        # We will extract the CLS token representation from the ViT's encoder
        self.backbone.heads = nn.Identity()
        
        # Optional: freeze backbone early layers to speed up training & protect representations
        # We freeze the first 6 encoder blocks of the 12 blocks in ViT-B
        for i, param in enumerate(self.backbone.parameters()):
            if i < 80: # Freeze early layers
                param.requires_grad = False

        # 2. Bidirectional Cross-Scale Attention Fusion
        self.cross_attn_20_40 = CrossScaleAttention(embed_dim, num_heads)
        self.cross_attn_40_20 = CrossScaleAttention(embed_dim, num_heads)
        
        # Projection layer after concatenating the two fused branches
        self.fusion_proj = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.LayerNorm(embed_dim)
        )
        
        # 3. Multi-Task Classification Heads
        # Head A: Malignancy (0 = Normal, 1 = Carcinoma)
        self.malignancy_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim // 2, 2) # 2 classes: normal vs malignant
        )
        
        # Head B: Carcinoma Subtype (0 = Adenocarcinoma, 1 = Squamous Cell Carcinoma)
        # (This loss will be masked out for normal samples)
        self.subtype_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim // 2, 2) # 2 classes: aca vs scc
        )
        
        # Head C: Ordinal Differentiation Level (CORAL - Consistent Ordinal Regression)
        # For K = 3 classes (Well=0, Moderate=1, Poor=2), we need K-1 = 2 binary classifiers:
        # Output 0: P(Grade >= Moderate)
        # Output 1: P(Grade >= Poor)
        self.ordinal_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim // 2, num_diff_classes - 1) # 2 binary outputs
        )

    def _extract_vit_tokens(self, x):
        # Reshape and permute the input tensor via ViT projection
        x = self.backbone._process_input(x) # [batch_size, 196, 768]
        n = x.shape[0]

        # Expand the class token to the full batch
        batch_class_token = self.backbone.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1) # [batch_size, 197, 768]

        # Pass through the transformer encoder blocks
        x = self.backbone.encoder(x) # [batch_size, 197, 768]
        return x

    def forward(self, img_20x, img_40x, return_attention=False):
        # img_20x: [batch_size, 3, 224, 224]
        # img_40x: [batch_size, 3, 224, 224]
        
        # 1. Feature Extraction via Shared ViT (extract all 197 tokens)
        feat_20x_seq = self._extract_vit_tokens(img_20x) # [batch_size, 197, 768]
        feat_40x_seq = self._extract_vit_tokens(img_40x) # [batch_size, 197, 768]
        
        # 2. Bidirectional Cross-Scale Attention
        if return_attention:
            # 20x queries 40x (macro-structure guided by cytologic details)
            f_20_40, attn_20_40 = self.cross_attn_20_40(feat_20x_seq, feat_40x_seq, return_attn=True) # [batch_size, 197, 768]
            # 40x queries 20x (micro-cytology guided by architectural context)
            f_40_20, attn_40_20 = self.cross_attn_40_20(feat_40x_seq, feat_20x_seq, return_attn=True) # [batch_size, 197, 768]
        else:
            f_20_40 = self.cross_attn_20_40(feat_20x_seq, feat_40x_seq)
            f_40_20 = self.cross_attn_40_20(feat_40x_seq, feat_20x_seq)
        
        # Concatenate and project at token level
        fused = torch.cat([f_20_40, f_40_20], dim=-1) # [batch_size, 197, 1536]
        fused_seq = self.fusion_proj(fused) # [batch_size, 197, 768]
        
        # Pool: Extract the CLS token representation at index 0
        fused_representation = fused_seq[:, 0] # [batch_size, 768]
        
        # 3. Forward through Multi-Task Heads
        logits_malignancy = self.malignancy_head(fused_representation) # [batch_size, 2]
        logits_subtype = self.subtype_head(fused_representation)       # [batch_size, 2]
        logits_ordinal = self.ordinal_head(fused_representation)       # [batch_size, 2]
        
        out = {
            'malignancy': logits_malignancy,
            'subtype': logits_subtype,
            'ordinal': logits_ordinal
        }
        
        if return_attention:
            out['attn_20_40'] = attn_20_40
            out['attn_40_20'] = attn_40_20
            
        return out


class CoralOrdinalLoss(nn.Module):
    """
    Computes the Consistent Ordinal Regression (CORAL) loss for PyTorch.
    For K classes, targets are converted into K-1 binary indicators.
    """
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, logits, targets):
        # logits: [batch_size, num_classes - 1] (here [batch_size, 2])
        # targets: [batch_size] containing integers in [0, 1, 2]
        
        batch_size, num_tasks = logits.size()
        
        # Convert ordinal targets to binary indicators
        # Target 0 (Well)     -> [0, 0]
        # Target 1 (Moderate) -> [1, 0]
        # Target 2 (Poor)     -> [1, 1]
        binary_targets = torch.zeros_like(logits)
        for i in range(num_tasks):
            binary_targets[:, i] = (targets > i).float()
            
        # Compute BCE loss for each ordinal task
        loss = self.bce(logits, binary_targets)
        return loss.mean()

def predict_ordinal_class(logits):
    """
    Helper function to convert CORAL ordinal logits into class predictions [0, 1, 2].
    Prediction = sum of binary decisions where sigmoid(logit) > 0.5.
    """
    probs = torch.sigmoid(logits)
    predictions = (probs > 0.5).sum(dim=-1)
    return predictions
