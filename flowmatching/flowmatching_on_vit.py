import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from typing import Optional, Tuple, List, Dict, Union
import sys
import numpy as np
from tqdm import tqdm
import logging
import copy
import traceback
from dataclasses import dataclass
from scipy.optimize import linear_sum_assignment
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import logging
from collections import defaultdict
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from sklearn.decomposition import PCA



# Set up device and logging
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.basicConfig(stream=sys.stdout, format='%(asctime)s %(levelname)s: %(message)s', level=logging.INFO, datefmt='%I:%M:%S')


# ======================== UTILITY CLASSES ========================
class Bunch:
    """Simple Bunch class for storing data"""
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

def count_parameters(model):
    """Count trainable parameters in a model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

@dataclass
class AttentionWeights:
    """Container for multi-head attention weights"""
    qkv_weight: torch.Tensor  # Combined QKV projection
    qkv_bias: Optional[torch.Tensor]
    proj_weight: torch.Tensor  # Output projection
    proj_bias: Optional[torch.Tensor]
    num_heads: int
    
    def split_heads(self):
        """Split QKV weights by heads"""
        d_model = self.qkv_weight.shape[1]
        head_dim = d_model // self.num_heads
        
        # Reshape QKV: [3*d_model, d_model] -> [3, num_heads, head_dim, d_model]
        qkv = self.qkv_weight.reshape(3, self.num_heads, head_dim, d_model)
        q_weights = qkv[0]  # [num_heads, head_dim, d_model]
        k_weights = qkv[1]
        v_weights = qkv[2]
        
        return q_weights, k_weights, v_weights

@dataclass
class TransformerBlockWeights:
    """Container for transformer block weights"""
    attention: AttentionWeights
    norm1_weight: torch.Tensor
    norm1_bias: torch.Tensor
    mlp_weights: Tuple[torch.Tensor, ...]  # MLP layer weights
    mlp_biases: Tuple[torch.Tensor, ...]
    norm2_weight: torch.Tensor
    norm2_bias: torch.Tensor

class VisionTransformerWeightSpace:
    """Weight space object for Vision Transformers"""
    
    def __init__(self, 
                 patch_embed_weight: torch.Tensor,
                 patch_embed_bias: Optional[torch.Tensor],
                 cls_token: torch.Tensor,
                 pos_embed: torch.Tensor,
                 blocks: List[TransformerBlockWeights],
                 norm_weight: torch.Tensor,
                 norm_bias: torch.Tensor,
                 head_weight: torch.Tensor,
                 head_bias: torch.Tensor):
        
        self.patch_embed_weight = patch_embed_weight
        self.patch_embed_bias = patch_embed_bias
        self.cls_token = cls_token
        self.pos_embed = pos_embed
        self.blocks = blocks
        self.norm_weight = norm_weight
        self.norm_bias = norm_bias
        self.head_weight = head_weight
        self.head_bias = head_bias
        
    @classmethod
    def from_vit_model(cls, model: nn.Module):
        """Extract weights from a ViT model"""
        blocks = []
        
        # Extract transformer blocks
        for block in model.blocks:
            # Multi-head attention weights
            attn = block.attn
            attention_weights = AttentionWeights(
                qkv_weight=attn.qkv.weight.data.clone(),
                qkv_bias=attn.qkv.bias.data.clone() if attn.qkv.bias is not None else None,
                proj_weight=attn.proj.weight.data.clone(),
                proj_bias=attn.proj.bias.data.clone() if attn.proj.bias is not None else None,
                num_heads=attn.num_heads
            )
            
            # MLP weights - iterate through MLP's children modules
            mlp_weights = []
            mlp_biases = []
            for name, layer in block.mlp.named_children():
                if hasattr(layer, 'weight'):
                    mlp_weights.append(layer.weight.data.clone())
                    if hasattr(layer, 'bias') and layer.bias is not None:
                        mlp_biases.append(layer.bias.data.clone())
            mlp_weights = tuple(mlp_weights)
            mlp_biases = tuple(mlp_biases)
            
            # Layer norms
            block_weights = TransformerBlockWeights(
                attention=attention_weights,
                norm1_weight=block.norm1.weight.data.clone(),
                norm1_bias=block.norm1.bias.data.clone(),
                mlp_weights=mlp_weights,
                mlp_biases=mlp_biases,
                norm2_weight=block.norm2.weight.data.clone(),
                norm2_bias=block.norm2.bias.data.clone()
            )
            blocks.append(block_weights)
        
        # Create weight space object
        return cls(
            patch_embed_weight=model.patch_embed.proj.weight.data.clone(),
            patch_embed_bias=model.patch_embed.proj.bias.data.clone() 
                            if model.patch_embed.proj.bias is not None else None,
            cls_token=model.cls_token.data.clone(),
            pos_embed=model.pos_embed.data.clone(),
            blocks=blocks,
            norm_weight=model.norm.weight.data.clone(),
            norm_bias=model.norm.bias.data.clone(),
            head_weight=model.head.weight.data.clone(),
            head_bias=model.head.bias.data.clone()
        )
    
    def apply_to_model(self, model: nn.Module):
        """Apply weights to a ViT model"""
        with torch.no_grad():
            # Patch embedding
            model.patch_embed.proj.weight.data.copy_(self.patch_embed_weight)
            if self.patch_embed_bias is not None:
                model.patch_embed.proj.bias.data.copy_(self.patch_embed_bias)
            
            # Tokens and embeddings
            model.cls_token.data.copy_(self.cls_token)
            model.pos_embed.data.copy_(self.pos_embed)
            
            # Transformer blocks
            for block, block_weights in zip(model.blocks, self.blocks):
                # Attention
                attn = block.attn
                attn.qkv.weight.data.copy_(block_weights.attention.qkv_weight)
                if block_weights.attention.qkv_bias is not None:
                    attn.qkv.bias.data.copy_(block_weights.attention.qkv_bias)
                attn.proj.weight.data.copy_(block_weights.attention.proj_weight)
                if block_weights.attention.proj_bias is not None:
                    attn.proj.bias.data.copy_(block_weights.attention.proj_bias)
                
                # Layer norms
                block.norm1.weight.data.copy_(block_weights.norm1_weight)
                block.norm1.bias.data.copy_(block_weights.norm1_bias)
                block.norm2.weight.data.copy_(block_weights.norm2_weight)
                block.norm2.bias.data.copy_(block_weights.norm2_bias)
                
                # MLP - iterate through MLP's children modules
                mlp_layers = [layer for name, layer in block.mlp.named_children() 
                             if hasattr(layer, 'weight')]
                for layer, weight in zip(mlp_layers, block_weights.mlp_weights):
                    layer.weight.data.copy_(weight)
                    
                # Handle biases separately since not all layers may have them
                mlp_bias_idx = 0
                for name, layer in block.mlp.named_children():
                    if hasattr(layer, 'bias') and layer.bias is not None:
                        if mlp_bias_idx < len(block_weights.mlp_biases):
                            layer.bias.data.copy_(block_weights.mlp_biases[mlp_bias_idx])
                            mlp_bias_idx += 1
            
            # Final norm and head
            model.norm.weight.data.copy_(self.norm_weight)
            model.norm.bias.data.copy_(self.norm_bias)
            model.head.weight.data.copy_(self.head_weight)
            model.head.bias.data.copy_(self.head_bias)
    
    def flatten(self, device=None) -> torch.Tensor:
        """Flatten all weights into a single vector"""
        all_params = []
        
        # Patch embedding
        all_params.append(self.patch_embed_weight.flatten())
        if self.patch_embed_bias is not None:
            all_params.append(self.patch_embed_bias.flatten())
        
        # Tokens and embeddings
        all_params.append(self.cls_token.flatten())
        all_params.append(self.pos_embed.flatten())
        
        # Transformer blocks
        for block in self.blocks:
            # Attention
            all_params.append(block.attention.qkv_weight.flatten())
            if block.attention.qkv_bias is not None:
                all_params.append(block.attention.qkv_bias.flatten())
            all_params.append(block.attention.proj_weight.flatten())
            if block.attention.proj_bias is not None:
                all_params.append(block.attention.proj_bias.flatten())
            
            # Norms
            all_params.append(block.norm1_weight.flatten())
            all_params.append(block.norm1_bias.flatten())
            all_params.append(block.norm2_weight.flatten())
            all_params.append(block.norm2_bias.flatten())
            
            # MLP
            for w in block.mlp_weights:
                all_params.append(w.flatten())
            for b in block.mlp_biases:
                all_params.append(b.flatten())
        
        # Final norm and head
        all_params.append(self.norm_weight.flatten())
        all_params.append(self.norm_bias.flatten())
        all_params.append(self.head_weight.flatten())
        all_params.append(self.head_bias.flatten())
        
        flat = torch.cat(all_params)
        if device:
            flat = flat.to(device)
        return flat
    
    @classmethod
    def from_flat(cls, flat_tensor, reference_ws, device=None):
        """Reconstruct VisionTransformerWeightSpace from flattened weights"""
        if device is None:
            device = flat_tensor.device
            
        # Get all parameter shapes from reference
        param_shapes = []
        param_types = []
        
        # Patch embedding
        param_shapes.append(reference_ws.patch_embed_weight.shape)
        param_types.append('patch_embed_weight')
        
        if reference_ws.patch_embed_bias is not None:
            param_shapes.append(reference_ws.patch_embed_bias.shape)
            param_types.append('patch_embed_bias')
        
        # CLS token and pos embed
        param_shapes.append(reference_ws.cls_token.shape)
        param_types.append('cls_token')
        
        param_shapes.append(reference_ws.pos_embed.shape)
        param_types.append('pos_embed')
        
        # Transformer blocks
        for block_idx, block in enumerate(reference_ws.blocks):
            # Attention weights
            param_shapes.append(block.attention.qkv_weight.shape)
            param_types.append(f'block_{block_idx}_attn_qkv_weight')
            
            if block.attention.qkv_bias is not None:
                param_shapes.append(block.attention.qkv_bias.shape)
                param_types.append(f'block_{block_idx}_attn_qkv_bias')
            
            param_shapes.append(block.attention.proj_weight.shape)
            param_types.append(f'block_{block_idx}_attn_proj_weight')
            
            if block.attention.proj_bias is not None:
                param_shapes.append(block.attention.proj_bias.shape)
                param_types.append(f'block_{block_idx}_attn_proj_bias')
            
            # Layer norms
            param_shapes.append(block.norm1_weight.shape)
            param_types.append(f'block_{block_idx}_norm1_weight')
            
            param_shapes.append(block.norm1_bias.shape)
            param_types.append(f'block_{block_idx}_norm1_bias')
            
            param_shapes.append(block.norm2_weight.shape)
            param_types.append(f'block_{block_idx}_norm2_weight')
            
            param_shapes.append(block.norm2_bias.shape)
            param_types.append(f'block_{block_idx}_norm2_bias')
            
            # MLP weights
            for mlp_idx, mlp_weight in enumerate(block.mlp_weights):
                param_shapes.append(mlp_weight.shape)
                param_types.append(f'block_{block_idx}_mlp_weight_{mlp_idx}')
            
            # MLP biases
            for mlp_idx, mlp_bias in enumerate(block.mlp_biases):
                param_shapes.append(mlp_bias.shape)
                param_types.append(f'block_{block_idx}_mlp_bias_{mlp_idx}')
        
        # Final norm and head
        param_shapes.append(reference_ws.norm_weight.shape)
        param_types.append('norm_weight')
        
        param_shapes.append(reference_ws.norm_bias.shape)
        param_types.append('norm_bias')
        
        param_shapes.append(reference_ws.head_weight.shape)
        param_types.append('head_weight')
        
        param_shapes.append(reference_ws.head_bias.shape)
        param_types.append('head_bias')
        
        # Split flat tensor according to shapes
        sizes = [np.prod(shape) for shape in param_shapes]
        parts = []
        start = 0
        
        for size in sizes:
            parts.append(flat_tensor[start:start+size])
            start += size
        
        # Reconstruct parameters
        reconstructed_params = {}
        for i, (shape, param_type) in enumerate(zip(param_shapes, param_types)):
            reconstructed_params[param_type] = parts[i].reshape(shape).to(device)
        
        # Build the blocks
        reconstructed_blocks = []
        num_blocks = len(reference_ws.blocks)
        
        for block_idx in range(num_blocks):
            # Reconstruct attention weights
            qkv_weight = reconstructed_params[f'block_{block_idx}_attn_qkv_weight']
            qkv_bias = reconstructed_params.get(f'block_{block_idx}_attn_qkv_bias', None)
            proj_weight = reconstructed_params[f'block_{block_idx}_attn_proj_weight']  
            proj_bias = reconstructed_params.get(f'block_{block_idx}_attn_proj_bias', None)
            
            attention = AttentionWeights(
                qkv_weight=qkv_weight,
                qkv_bias=qkv_bias,
                proj_weight=proj_weight,
                proj_bias=proj_bias,
                num_heads=reference_ws.blocks[block_idx].attention.num_heads
            )
            
            # Reconstruct MLP weights
            mlp_weights = []
            mlp_biases = []
            
            mlp_weight_idx = 0
            while f'block_{block_idx}_mlp_weight_{mlp_weight_idx}' in reconstructed_params:
                mlp_weights.append(reconstructed_params[f'block_{block_idx}_mlp_weight_{mlp_weight_idx}'])
                mlp_weight_idx += 1
            
            mlp_bias_idx = 0
            while f'block_{block_idx}_mlp_bias_{mlp_bias_idx}' in reconstructed_params:
                mlp_biases.append(reconstructed_params[f'block_{block_idx}_mlp_bias_{mlp_bias_idx}'])
                mlp_bias_idx += 1
            
            # Create block
            block = TransformerBlockWeights(
                attention=attention,
                norm1_weight=reconstructed_params[f'block_{block_idx}_norm1_weight'],
                norm1_bias=reconstructed_params[f'block_{block_idx}_norm1_bias'],
                mlp_weights=tuple(mlp_weights),
                mlp_biases=tuple(mlp_biases),
                norm2_weight=reconstructed_params[f'block_{block_idx}_norm2_weight'],
                norm2_bias=reconstructed_params[f'block_{block_idx}_norm2_bias']
            )
            
            reconstructed_blocks.append(block)
        
        # Create the full weight space object
        return cls(
            patch_embed_weight=reconstructed_params['patch_embed_weight'],
            patch_embed_bias=reconstructed_params.get('patch_embed_bias', None),
            cls_token=reconstructed_params['cls_token'],
            pos_embed=reconstructed_params['pos_embed'],
            blocks=reconstructed_blocks,
            norm_weight=reconstructed_params['norm_weight'], 
            norm_bias=reconstructed_params['norm_bias'],
            head_weight=reconstructed_params['head_weight'],
            head_bias=reconstructed_params['head_bias']
        )

# ======================== VIT MODEL DEFINITION ========================
class MultiHeadAttention(nn.Module):
    """Multi-head attention module"""
    
    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        
        return x

class MLP(nn.Module):
    """MLP module"""
    
    def __init__(self, in_features: int, hidden_features: Optional[int] = None, 
                 out_features: Optional[int] = None, dropout: float = 0.1):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features * 4
        
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

class TransformerBlock(nn.Module):
    """Transformer block with attention and MLP"""
    
    def __init__(self, dim: int, num_heads: int = 8, mlp_ratio: float = 4.0, 
                 dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadAttention(dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim * mlp_ratio), dropout=dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class PatchEmbed(nn.Module):
    """Image to patch embedding"""
    
    def __init__(self, img_size: int = 32, patch_size: int = 4, 
                 in_chans: int = 3, embed_dim: int = 192):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        
        self.proj = nn.Conv2d(in_chans, embed_dim, 
                             kernel_size=patch_size, stride=patch_size)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x

class VisionTransformer(nn.Module):
    """Simple Vision Transformer for CIFAR-10"""
    
    def __init__(self, 
                 img_size: int = 32,
                 patch_size: int = 4,
                 in_chans: int = 3,
                 num_classes: int = 10,
                 embed_dim: int = 512,
                 depth: int = 8,
                 num_heads: int = 8,
                 mlp_ratio: float = 4.0,
                 dropout: float = 0.1):
        super().__init__()
        
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=dropout)
        
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        
        self._init_weights()
        
    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
                
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        
        x = self.patch_embed(x)
        
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        for block in self.blocks:
            x = block(x)
        
        x = self.norm(x)
        x = x[:, 0]
        x = self.head(x)
        
        return x

def create_vit_small(num_classes: int = 10, **kwargs) -> VisionTransformer:
    """Create a small ViT suitable for CIFAR-10"""
    
    return VisionTransformer(
        img_size=32,
        patch_size=4,
        embed_dim=192,
        depth=6,
        num_heads=3,
        mlp_ratio=4,
        num_classes=num_classes,
        dropout=0.1
    )

# ======================== DATA LOADING ========================
def load_cifar10(batch_size=128):

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    test_set = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2)

    return test_loader

def evaluate(model, device):
    """Evaluate model on test set"""
    test_loader = load_cifar10(batch_size=128)
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            pred = out.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return (correct / total)*100

def print_stats(models, device):
    accuracies = []
    for i, model in enumerate(models):
        acc = evaluate(model, device)
        accuracies.append(acc)

    accuracies = np.array(accuracies)
    mean = accuracies.mean()
    std = accuracies.std()
    min_acc = accuracies.min()
    max_acc = accuracies.max()

    logging.info("\n=== Summary ===")
    logging.info(f"Average Accuracy: {mean:.2f}% ± {std:.2f}%")
    logging.info(f"Min Accuracy: {min_acc:.2f}%")
    logging.info(f"Max Accuracy: {max_acc:.2f}%")

class PermutationSpec:
    """Specification for permutations applied throughout the network"""
    
    def __init__(self, num_blocks: int):
        self.num_blocks = num_blocks
        # Store permutations for each layer
        self.patch_embed_perm = None
        self.block_perms = []  # List of dicts with attention and mlp permutations
        for _ in range(num_blocks):
            self.block_perms.append({
                'attention_in': None,  # Input permutation to attention
                'attention_out': None,  # Output permutation from attention
                'mlp1': None,  # First MLP layer permutation
                'mlp2': None,  # Second MLP layer permutation  
            })
        self.head_perm = None  # Final classification head
        
    def set_block_perm(self, block_idx: int, perm_type: str, perm: torch.Tensor):
        """Set a specific permutation for a block"""
        self.block_perms[block_idx][perm_type] = perm


class TransFusionMatcher:
    """
    Weight matching using TransFusion approach:
    - Two-level permutation for attention heads
    - Handling of residual connections
    - Iterative refinement
    """
    
    def __init__(self, num_iterations: int = 5, epsilon: float = 1e-8):
        self.num_iterations = num_iterations
        self.epsilon = epsilon
        
    def compute_spectral_distance(self, 
                                 weight1: torch.Tensor, 
                                 weight2: torch.Tensor) -> float:
        """
        Compute permutation-invariant distance using singular values
        as described in TransFusion paper
        """
        try:
            _, s1, _ = torch.svd(weight1)
            _, s2, _ = torch.svd(weight2)
        except:
            # Fallback to numpy if torch SVD fails
            _, s1, _ = np.linalg.svd(weight1.numpy())
            _, s2, _ = np.linalg.svd(weight2.numpy())
            s1 = torch.tensor(s1)
            s2 = torch.tensor(s2)
        
        max_len = max(len(s1), len(s2))
        if len(s1) < max_len:
            s1 = torch.cat([s1, torch.zeros(max_len - len(s1))])
        if len(s2) < max_len:
            s2 = torch.cat([s2, torch.zeros(max_len - len(s2))])
        
        return torch.norm(s1 - s2).item()
    
    def compose_attention_permutation(self,
                                     inter_head_perm: torch.Tensor,
                                     intra_head_perms: List[torch.Tensor],
                                     d_model: int,
                                     num_heads: int) -> torch.Tensor:
        """
        Compose inter and intra head permutations into a single block diagonal matrix
        Following Theorem 3.1 from the paper
        """
        head_dim = d_model // num_heads
        
        # Create block diagonal permutation matrix
        P_attn = torch.zeros(d_model, d_model)
        
        for i in range(num_heads):
            # Find which head maps to position i
            j = torch.argmax(inter_head_perm[:, i]).item()
            
            # Get the intra-head permutation for this head
            P_intra = intra_head_perms[j] if j < len(intra_head_perms) else torch.eye(head_dim)
            
            # Place in the block diagonal
            start_i = i * head_dim
            end_i = (i + 1) * head_dim
            start_j = j * head_dim
            end_j = (j + 1) * head_dim
            
            # Apply the permutation
            P_attn[start_i:end_i, start_j:end_j] = P_intra
        
        return P_attn
    
    def match_attention_heads(self,
                            attn1: AttentionWeights,
                            attn2: AttentionWeights) -> Tuple[torch.Tensor, List[torch.Tensor], torch.Tensor]:
        """
        Two-level matching for attention heads:
        1. Inter-head alignment using spectral distance
        2. Intra-head alignment using weight matching
        Returns: inter_head_perm, intra_head_perms, composed_perm
        """
        # Split heads
        q1, k1, v1 = attn1.split_heads()
        q2, k2, v2 = attn2.split_heads()
        
        num_heads = attn1.num_heads
        d_model = attn1.qkv_weight.shape[1]
        head_dim = d_model // num_heads
        
        distance_matrix = torch.zeros(num_heads, num_heads)
        
        for i in range(num_heads):
            for j in range(num_heads):
                dist_q = self.compute_spectral_distance(q1[i], q2[j])
                dist_k = self.compute_spectral_distance(k1[i], k2[j])
                dist_v = self.compute_spectral_distance(v1[i], v2[j])
                distance_matrix[i, j] = dist_q + dist_k + dist_v
        
        row_ind, col_ind = linear_sum_assignment(distance_matrix.detach().cpu().numpy())

        inter_head_perm = torch.zeros(num_heads, num_heads)
        inter_head_perm[row_ind, col_ind] = 1.0
        
        intra_head_perms = []
        for i, j in zip(row_ind, col_ind):
            cost_q = -torch.mm(q2[j], q1[i].t())
            cost_k = -torch.mm(k2[j], k1[i].t())
            cost_v = -torch.mm(v2[j], v1[i].t())
            cost_matrix = (cost_q + cost_k + cost_v) / 3.0
            
            row_ind_intra, col_ind_intra = linear_sum_assignment(cost_matrix.detach().cpu().numpy())
            
            perm = torch.zeros(head_dim, head_dim)
            perm[row_ind_intra, col_ind_intra] = 1.0
            intra_head_perms.append(perm)
        
        composed_perm = self.compose_attention_permutation(
            inter_head_perm, intra_head_perms, d_model, num_heads
        )
        
        return inter_head_perm, intra_head_perms, composed_perm
    
    def match_mlp_layer(self, 
                    weight1: torch.Tensor, 
                    weight2: torch.Tensor,
                    prev_perm: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Match MLP layers using Hungarian algorithm
        Following Git Re-Basin approach but accounting for previous permutation
        """
        device = weight1.device 
    
        if prev_perm is not None:
            prev_perm = prev_perm.to(device) 
            if prev_perm.shape[0] == weight1.shape[1]:
                weight1_permuted = torch.mm(weight1, prev_perm.t())
            else:
                weight1_permuted = weight1
        else:
            weight1_permuted = weight1
    
        cost_matrix = -torch.mm(weight2, weight1_permuted.t())
    
        row_ind, col_ind = linear_sum_assignment(cost_matrix.detach().cpu().numpy())
    
        n = weight1.shape[0]
        perm = torch.zeros(n, n, device=device)
        perm[row_ind, col_ind] = 1.0
    
        return perm

    
    def apply_permutation_to_weights(self,
                                 weights: VisionTransformerWeightSpace,
                                 perm_spec: PermutationSpec) -> VisionTransformerWeightSpace:
        """
        Apply computed permutations to weight space maintaining functional equivalence
        Key insight: When permuting layer l with P_l, must apply P_l^T to layer l+1
        """
        result = copy.deepcopy(weights)
        
        prev_output_perm = None
        
        for block_idx, (source_block, result_block) in enumerate(
            zip(weights.blocks, result.blocks)):
            
            block_perm = perm_spec.block_perms[block_idx]
            d_model = source_block.attention.qkv_weight.shape[1]
    
            device = result_block.attention.qkv_weight.device
            
            if block_perm['attention_out'] is not None:
                P_attn = block_perm['attention_out'].to(device)
                
                P_attn_expanded = torch.block_diag(P_attn, P_attn, P_attn).to(device)
                
                if prev_output_perm is not None:
                    result_block.attention.qkv_weight = torch.mm(
                        torch.mm(P_attn_expanded, result_block.attention.qkv_weight),
                        prev_output_perm.t()
                    )
                else:
                    result_block.attention.qkv_weight = torch.mm(
                        P_attn_expanded, result_block.attention.qkv_weight
                    )
                
                if result_block.attention.qkv_bias is not None:
                    result_block.attention.qkv_bias = torch.mv(
                        P_attn_expanded, result_block.attention.qkv_bias
                    )
                
                result_block.attention.proj_weight = torch.mm(
                    result_block.attention.proj_weight, P_attn.t()
                )
            
            if len(result_block.mlp_weights) >= 2 and block_perm['mlp1'] is not None:
                P_mlp1 = block_perm['mlp1'].to(device)
                
                result_block.mlp_weights = list(result_block.mlp_weights)
                if prev_output_perm is not None:
                    result_block.mlp_weights[0] = torch.mm(
                        torch.mm(P_mlp1, result_block.mlp_weights[0]),
                        prev_output_perm.t()
                    )
                else:
                    result_block.mlp_weights[0] = torch.mm(P_mlp1, result_block.mlp_weights[0])
                
                if len(result_block.mlp_biases) > 0:
                    result_block.mlp_biases = list(result_block.mlp_biases)
                    result_block.mlp_biases[0] = torch.mv(P_mlp1, result_block.mlp_biases[0])
                
                result_block.mlp_weights[1] = torch.mm(
                    result_block.mlp_weights[1], P_mlp1.t()
                )
                
                result_block.mlp_weights = tuple(result_block.mlp_weights)
                result_block.mlp_biases = tuple(result_block.mlp_biases)
                
                prev_output_perm = torch.eye(d_model, device=device)
            else:
                prev_output_perm = torch.eye(d_model, device=device)
        
        return result

    
    def canonicalize_model(self,
                          models: List[VisionTransformerWeightSpace],
                          reference_idx: int = 0) -> List[VisionTransformerWeightSpace]:
        """
        Canonicalize multiple models using one as reference
        Implements Algorithm 1 from the paper with iterative refinement
        """
        reference = models[reference_idx]
        canonicalized = []
        
        for i, model in enumerate(models):
            if i == reference_idx:
                canonicalized.append(reference)
            else:
                
                if i == 1:
                    orig_block0_attn = model.blocks[0].attention.qkv_weight.clone()
                    orig_block0_mlp0 = model.blocks[0].mlp_weights[0].clone() if len(model.blocks[0].mlp_weights) > 0 else None
                
                perm_spec = PermutationSpec(len(model.blocks))
                current_model = copy.deepcopy(model)
                
                for iteration in range(self.num_iterations):
                    
                    current_dim_perm = None
                    
                    for block_idx in range(len(current_model.blocks)):
                        current_block = current_model.blocks[block_idx]
                        reference_block = reference.blocks[block_idx]
                        
                        d_model = current_block.attention.qkv_weight.shape[1]
                        
                        inter_perm, intra_perms, composed_perm = self.match_attention_heads(
                            current_block.attention, reference_block.attention
                        )
                        perm_spec.set_block_perm(block_idx, 'attention_out', composed_perm)
                        
                        if len(current_block.mlp_weights) >= 1:
                            mlp1_perm = self.match_mlp_layer(
                                current_block.mlp_weights[0],
                                reference_block.mlp_weights[0],
                                prev_perm=current_dim_perm
                            )
                            perm_spec.set_block_perm(block_idx, 'mlp1', mlp1_perm)
                            
                            current_dim_perm = torch.eye(d_model)
                    
                    current_model = self.apply_permutation_to_weights(model, perm_spec)
                
                if i == 1:
                    attn_diff = torch.norm(current_model.blocks[0].attention.qkv_weight - orig_block0_attn).item()
                    
                    if orig_block0_mlp0 is not None:
                        mlp_diff = torch.norm(current_model.blocks[0].mlp_weights[0] - orig_block0_mlp0).item()
                                    
                orig_flat = model.flatten()
                canon_flat = current_model.flatten()
                weight_change = torch.norm(canon_flat - orig_flat).item()
                
                canonicalized.append(current_model)
        
        return canonicalized

class FlowMatching:
    def __init__(
        self,
        sourceloader,
        targetloader,
        model,
        mode="velocity",
        t_dist="uniform",
        device=None,
        normalize_pred=False,
        geometric=False,
    ):
        # Device
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.sourceloader = sourceloader
        self.targetloader = targetloader
        self.model = model.to(self.device)
        self.mode = mode
        self.t_dist = t_dist
        self.sigma = 0.001
        self.normalize_pred = normalize_pred
        self.geometric = geometric

        # Metrics tracking
        self.metrics = {"train_loss": [], "time": [], "grad_norm": [], "flow_norm": [], "true_norm": []}
        self.best_loss = float('inf')
        self.best_model_state = None

        self.input_dim = None

    # ---------------- Sampling ----------------
    def sample_from_loader(self, loader):
        """Sample a batch from dataloader"""
        try:
            if not hasattr(loader, '_iterator') or loader._iterator is None:
                loader._iterator = iter(loader)
            try:
                batch = next(loader._iterator)
            except StopIteration:
                loader._iterator = iter(loader)
                batch = next(loader._iterator)
            return batch[0].to(self.device)
        except Exception as e:
            logging.info(f"Error sampling from loader: {str(e)}")
            if hasattr(loader.dataset, '__getitem__'):
                dummy = loader.dataset[0][0]
                return torch.zeros(loader.batch_size, *dummy.shape, device=self.device)
            return torch.zeros(loader.batch_size, 1, device=self.device)

    def sample_time_and_flow(self):
        """Sample time t and flow (for velocity or target mode)"""
        x0 = self.sample_from_loader(self.sourceloader)
        x1 = self.sample_from_loader(self.targetloader)
        batch_size = min(x0.size(0), x1.size(0))
        x0, x1 = x0[:batch_size], x1[:batch_size]

        if self.t_dist == "beta":
            alpha, beta_param = 2.0, 5.0
            t = torch.distributions.Beta(alpha, beta_param).sample((batch_size,)).to(self.device)
        else:
            t = torch.rand(batch_size, device=self.device)

        t_pad = t.view(-1, *([1] * (x0.dim() - 1)))
        mu_t = (1 - t_pad) * x0 + t_pad * x1
        epsilon = torch.randn_like(x0) * self.sigma
        xt = mu_t + epsilon
        ut = x1 - x0

        return Bunch(t=t.unsqueeze(-1), x0=x0, xt=xt, x1=x1, ut=ut, eps=epsilon, batch_size=batch_size)

    def forward(self, flow):
        flow_pred = self.model(flow.xt, flow.t)
        return None, flow_pred

    def loss_fn(self, flow_pred, flow):
        if self.mode == "target":
            l_flow = torch.mean((flow_pred.squeeze() - flow.x1) ** 2)
        else:
            l_flow = torch.mean((flow_pred.squeeze() - flow.ut) ** 2)
        return None, l_flow

    def vector_field(self, xt, t):
        """Compute vector field at point xt and time t"""
        _, pred = self.forward(Bunch(xt=xt, t=t, batch_size=xt.size(0)))
        return pred if self.mode == "velocity" else pred - xt
        
    def train(self, n_iters=10, optimizer=None, scheduler=None, sigma=0.001, patience=1e99, 
              log_freq=5, accum_steps=4, clip_grad=1.0):
        """Train the flow model with gradient accumulation (no mixed precision)"""
        self.sigma = sigma
        self.metrics = {"train_loss": [], "time": [], "grad_norm": [], "flow_norm": [], "true_norm": []}
        last_loss = 1e99
        patience_count = 0
    
        pbar = tqdm(range(n_iters), desc="Training steps")
        optimizer.zero_grad()
        accum_count = 0
    
        for i in pbar:
            try:
                flow = self.sample_time_and_flow()
    
                # Forward pass
                _, flow_pred = self.forward(flow)
                _, loss = self.loss_fn(flow_pred, flow)
    
                if torch.isnan(loss) or torch.isinf(loss):
                    logging.info(f"Skipping step {i} due to invalid loss: {loss.item()}")
                    continue
    
                # Scale loss for gradient accumulation
                loss_scaled = loss / accum_steps
                loss_scaled.backward()
                accum_count += 1
    
                # Step if enough accumulation or last iteration
                if accum_count == accum_steps or i == n_iters - 1:
                    if clip_grad is not None:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip_grad)
    
                    optimizer.step()
                    optimizer.zero_grad()
                    accum_count = 0
    
                # Save best model
                if loss.item() < self.best_loss:
                    self.best_loss = loss.item()
                    self.best_model_state = {k: v.clone() for k, v in self.model.state_dict().items()}
    
                # Early stopping
                if loss.item() > last_loss:
                    patience_count += 1
                    if patience_count >= patience:
                        logging.info(f"Early stopping at iteration {i}")
                        break
                else:
                    patience_count = 0
    
                last_loss = loss.item()
    
                # Logging
                if i % log_freq == 0:
                    train_loss_val = loss.item()
                    true_tensor = flow.ut if self.mode == "velocity" else flow.x1
                    grad_norm = self.get_grad_norm()
                    self.metrics["train_loss"].append(train_loss_val)
                    self.metrics["flow_norm"].append(flow_pred.norm(p=2, dim=1).mean().item())
                    self.metrics["time"].append(flow.t.mean().item())
                    self.metrics["true_norm"].append(true_tensor.norm(p=2, dim=1).mean().item())
                    self.metrics["grad_norm"].append(grad_norm)
    
                    pbar.set_description(f"Iters [loss {train_loss_val:.6f}, ∇ norm {grad_norm:.6f}]")
    
            except Exception as e:
                logging.info(f"Error during training iteration {i}: {str(e)}")
                continue

    def get_grad_norm(self):
        total = 0
        for p in self.model.parameters():
            if p.grad is not None:
                total += p.grad.detach().norm(2).item() ** 2
        return total ** 0.5

    def map(self, x0, n_steps=50, return_traj=False, method="euler"):
        if self.best_model_state is not None:
            current_state = {k: v.clone() for k, v in self.model.state_dict().items()}
            self.model.load_state_dict(self.best_model_state)

        self.model.eval()
        batch_size, flat_dim = x0.size()
        traj = [x0.detach().clone()] if return_traj else None
        xt = x0.clone()
        times = torch.linspace(0, 1, n_steps, device=self.device)
        dt = times[1] - times[0]

        for i, t in enumerate(times[:-1]):
            with torch.no_grad():
                t_tensor = torch.ones(batch_size, 1, device=self.device) * t
                pred = self.model(xt, t_tensor)
                if pred.dim() > 2: pred = pred.squeeze(-1)
                vt = pred if self.mode == "velocity" else pred - xt
                if method == "euler":
                    xt = xt + vt * dt
                elif method == "rk4":
                    # RK4 steps
                    k1 = vt
                    k2 = self.model(xt + 0.5 * dt * k1, t_tensor + 0.5 * dt)
                    if k2.dim() > 2: k2 = k2.squeeze(-1)
                    k2 = k2 if self.mode == "velocity" else k2 - (xt + 0.5 * dt * k1)
                    k3 = self.model(xt + 0.5 * dt * k2, t_tensor + 0.5 * dt)
                    if k3.dim() > 2: k3 = k3.squeeze(-1)
                    k3 = k3 if self.mode == "velocity" else k3 - (xt + 0.5 * dt * k2)
                    k4 = self.model(xt + dt * k3, t_tensor + dt)
                    if k4.dim() > 2: k4 = k4.squeeze(-1)
                    k4 = k4 if self.mode == "velocity" else k4 - (xt + dt * k3)
                    xt = xt + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

                if return_traj:
                    traj.append(xt.detach().clone())

        if self.best_model_state is not None:
            self.model.load_state_dict(current_state)
        self.model.train()
        return traj if return_traj else xt

    def generate_weights(self, n_samples=10, source_noise_std=0.001, **map_kwargs):
        assert self.input_dim is not None, "Set `self.input_dim` before generating weights."
        source_samples = torch.randn(n_samples, self.input_dim, device=self.device) * source_noise_std
        return self.map(source_samples, **map_kwargs)

    def plot_metrics(self):
        labels = list(self.metrics.keys())
        lists = list(self.metrics.values())
        n = len(lists)
        fig, axs = plt.subplots(1, n, figsize=(3 * n, 3))
        for i, (label, lst) in enumerate(zip(labels, lists)):
            axs[i].plot(lst)
            axs[i].grid()
            axs[i].title.set_text(label)
            if label == "train_loss":
                axs[i].set_yscale("log")
        plt.tight_layout()
        plt.show()


class VisionTransformerFlowModel(nn.Module):
    """Flow model for ViT weight spaces"""
    
    def __init__(self, input_dim, time_embed_dim=64, hidden_dim = 256):
        super().__init__()
        self.input_dim = input_dim
        self.time_embed_dim = time_embed_dim
        
        self.time_embed = nn.Sequential(
            nn.Linear(1, time_embed_dim),
            nn.GELU(),
            nn.Linear(time_embed_dim, time_embed_dim)
        )
        
        self.net = nn.Sequential(
            nn.Linear(input_dim + time_embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.LayerNorm(hidden_dim//2), 
            nn.GELU(),
            nn.Dropout(0.1),
            
            nn.Linear(hidden_dim//2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            
            nn.Linear(hidden_dim, input_dim)
        )
        
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
    
    def forward(self, x, t):
        t_embed = self.time_embed(t)
        combined = torch.cat([x, t_embed], dim=-1)
        return self.net(combined)

def get_permuted_models_data(ref_point=0, model_dir="../cifar10_vit_small_patch4_models", 
                           num_models=100, device=None):
    """Load and align ViT models using rebasin"""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    ref_model = create_vit_small()
    ref_model_path = f"{model_dir}/vit_{ref_point}.pt"
    
    try:
        ref_model.load_state_dict(torch.load(ref_model_path, map_location=device))
        ref_model = ref_model.to(device)
        logging.info(f"Loaded reference model from {ref_model_path}")
    except Exception as e:
        logging.error(f"Failed to load reference model: {e}")
        raise e
    
    ref_ws = VisionTransformerWeightSpace.from_vit_model(ref_model)
    weight_space_objects = [ref_ws]
    org_weight_space_objects = [ref_ws]
    
    accuracies = []
    matcher = TransFusionMatcher(num_iterations=10)
    
    for i in tqdm(range(num_models), desc="Processing ViT models"):
        if i == ref_point:
            continue
        
        model_path = f"{model_dir}/vit_{i}.pt"
        if not os.path.exists(model_path):
            logging.warning(f"Skipping model {i} - file not found")
            continue
        
        model = create_vit_small()
        model.load_state_dict(torch.load(model_path, map_location=device))
        model = model.to(device)
        
        ws = VisionTransformerWeightSpace.from_vit_model(model)
        org_weight_space_objects.append(ws)
        # org_accuracy = evaluate(model, device)
        # accuracies.append(org_accuracy)
        
        canonicalized_list = matcher.canonicalize_model([ref_ws, ws], reference_idx=0)
        aligned_ws = canonicalized_list[1]
        
        weight_space_objects.append(aligned_ws)
        
        # new_model = create_vit_small().to(device)
        # aligned_ws.apply_to_model(new_model)
        # permuted_accuracy = evaluate(new_model, device)
        # logging.info(f"Org Accuracy:{org_accuracy} Permuted accuracy:{permuted_accuracy}")
        # torch.cuda.empty_cache()
    
    logging.info(f"Successfully processed {len(weight_space_objects)} ViT models")
    
    # logging.info("Orginal Models")
    # accuracies = np.array(accuracies)
    # mean = accuracies.mean()
    # std = accuracies.std()
    # min_acc = accuracies.min()
    # max_acc = accuracies.max()
    # logging.info("\n=== Summary ===")
    # logging.info(f"Average Accuracy: {mean:.2f}% ± {std:.2f}%")
    # logging.info(f"Min Accuracy: {min_acc:.2f}%")
    # logging.info(f"Max Accuracy: {max_acc:.2f}%")
    
    return ref_model, org_weight_space_objects, weight_space_objects

def generate_new_vit_models(cfm, reference_ws, vit_config, generated_flat, gen_method, n_samples=5):
    """Generate new ViT models using trained flow matching"""
    device = cfm.device
    
    logging.info(f"Generating {n_samples} new ViT models...")                
    generated_models = []
    
    for i in range(n_samples):
        generated_ws = VisionTransformerWeightSpace.from_flat(
            generated_flat[i], reference_ws, device
        )
        
        new_model = create_vit_small(**vit_config).to(device)
        generated_ws.apply_to_model(new_model)
        generated_models.append(new_model)
        del new_model
    
    return generated_models


def train_vit_flow_matching(vit_config=None, model_dir="../imagenet_vit_models", num_models=100):
    """ViT flow matching without PCA dimensionality reduction."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ref_model, org_weight_space_objects, weight_space_objects = get_permuted_models_data(
        model_dir=model_dir,
        num_models=num_models,
        device=device
    )
    reference_ws = weight_space_objects[0]

    for init_type in ["gaussian_0.01", "gaussian_0.001"]:
        for model_type in ["with_gitrebasin", "without_rebasin"]:
            models_to_use = weight_space_objects if model_type == "with_gitrebasin" else org_weight_space_objects
            logging.info(f"Processing {init_type} with {model_type}")
            logging.info("Flattening ViT weights on CPU...")

            flat_weights = [ws.flatten(device='cpu') for ws in models_to_use]
            X = torch.stack(flat_weights)
            flat_dim = X.shape[1]
            logging.info(f"ViT weight space dimension: {flat_dim:,}")

            del flat_weights
            torch.cuda.empty_cache()

            source_std = 0.01 if "0.01" in init_type else 0.001
            source_tensor = torch.randn(num_models, flat_dim, dtype=torch.float32) * source_std

            target_tensor = X
            source_dataset = TensorDataset(source_tensor)
            target_dataset = TensorDataset(target_tensor)

            sourceloader = DataLoader(source_dataset, batch_size=1, shuffle=True, pin_memory=True)
            targetloader = DataLoader(target_dataset, batch_size=1, shuffle=True, pin_memory=True)

            flow_model = VisionTransformerFlowModel(flat_dim).to(device)
            logging.info(f"Flow model parameters: {count_parameters(flow_model):,}")

            cfm = FlowMatching(
                sourceloader=sourceloader,
                targetloader=targetloader,
                model=flow_model,
                mode="velocity",
                t_dist="uniform",
                device=device
            )

            optimizer = torch.optim.AdamW(flow_model.parameters(), lr=5e-4, weight_decay=1e-5, betas=(0.9, 0.95))
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30000, eta_min=1e-6)

            cfm.train(n_iters=30000, optimizer=optimizer, scheduler=scheduler, sigma=0.001, patience=100, log_freq=10)

            logging.info("Generating new weights...")
            for gen_method in ["rk4"]:
                random_flat = torch.randn(num_models, flat_dim, dtype=torch.float32) * source_std

                generated_chunks = []
                chunk_size = 5 
                for i in range(0, num_models, chunk_size):
                    batch = random_flat[i:i+chunk_size].to(device)
                    gen = cfm.map(batch, n_steps=100, method=gen_method)
                    generated_chunks.append(gen.cpu())
                generated_flat = torch.cat(generated_chunks, dim=0)


                generated_models = generate_new_vit_models(
                    cfm, reference_ws, vit_config, generated_flat, gen_method, n_samples=25
                )

                logging.info(f"Init Type: {init_type}, Model Type: {model_type}, Generation Method: {gen_method}")
                print_stats(generated_models, device)

            del flow_model, cfm, random_flat, generated_flat, generated_models
            torch.cuda.empty_cache()


def main():
    """Main function demonstrating ViT flow matching"""
    logging.info("Starting ViT Flow Matching Pipeline...")
    
    vit_config = {
        'num_classes': 10,
        'embed_dim': 192,
        'depth': 6,
        'num_heads': 3,
        'mlp_ratio': 4.0,
        'dropout': 0.1
    }
    
    logging.info(f"Using ViT config: {vit_config}")
    
    logging.info("Testing model creation...")
    test_model = create_vit_small()
    
    train_vit_flow_matching(
        vit_config=vit_config,
        model_dir="../cifar10_vit_small_patch4_models",
        num_models=100
    )   
if __name__ == "__main__":
    logging.info("CIFAR-10 New ViT embed 384 time 64")
    main()
