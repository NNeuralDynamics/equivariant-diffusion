import torch
import torch.nn as nn
from typing import Tuple, List, Dict, Optional, Union
from dataclasses import dataclass
import numpy as np
from einops import rearrange


class WeightSpaceObject:
    """Base class for weight space objects (MLPs)"""
    def __init__(self, weights, biases):
        self.weights = weights if isinstance(weights, tuple) else tuple(weights)
        self.biases = biases if isinstance(biases, tuple) else tuple(biases)
        
    def flatten(self, device=None):
        """Flatten weights and biases into a single vector"""
        flat = torch.cat([w.flatten() for w in self.weights] + 
                          [b.flatten() for b in self.biases])
        if device:
            flat = flat.to(device)
        return flat
    
    @classmethod
    def from_flat(cls, flat, layers, device):
        """Create WeightSpaceObject from flattened vector"""
        sizes = []
        # Calculate sizes for weight matrices
        for i in range(len(layers) - 1):
            sizes.append(layers[i] * layers[i+1])  # Weight matrix
        # Calculate sizes for bias vectors
        for i in range(1, len(layers)):
            sizes.append(layers[i])  # Bias vector
            
        # Split flat tensor into parts
        parts = []
        start = 0
        for size in sizes:
            parts.append(flat[start:start+size])
            start += size
            
        # Reshape into weight matrices and bias vectors
        weights = []
        biases = []
        for i in range(len(layers) - 1):
            w_size = layers[i] * layers[i+1]
            weights.append(parts[i].reshape(layers[i+1], layers[i]))
            biases.append(parts[i + len(layers) - 1])
            
        return cls(weights, biases).to(device)
    
    def to(self, device):
        """Move weights and biases to specified device"""
        weights = tuple(w.to(device) for w in self.weights)
        biases = tuple(b.to(device) for b in self.biases)
        return WeightSpaceObject(weights, biases)
        
    def map(self, fn):
        new_weights = tuple(fn(w) for w in self.weights)
        new_biases = tuple(fn(b) for b in self.biases)
        return WeightSpaceObject(new_weights, new_biases)


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


class Bunch:
    """Simple Bunch class for storing data"""
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def count_parameters(model):
    """Count trainable parameters in a model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)