import os
import sys
import gc
import math
import copy
import logging
import traceback
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from scipy.optimize import linear_sum_assignment
from sklearn.utils.extmath import randomized_svd
from sklearn.metrics import r2_score
from transformers import BertTokenizer
from datasets import load_dataset
from typing import Optional, Tuple, List, Dict, Union, Any
from sklearn.utils.extmath import randomized_svd


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.basicConfig(
    stream=sys.stdout, 
    format='%(asctime)s %(levelname)s: %(message)s', 
    level=logging.INFO, 
    datefmt='%I:%M:%S'
)

class Bunch:
    """Simple Bunch class for storing data"""
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

def count_parameters(model):
    """Count trainable parameters in a model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def print_stats(models, device, test_loader=None):
    """Print statistics about generated models"""
    logging.info(f"Generated {len(models)} models")
    
    if test_loader is not None:
        for i, model in enumerate(models):
            r2 = evaluate_model(model, test_loader, device)
            logging.info(f"Model {i} R²: {r2:.4f}")
    
# ======================== WEIGHT SPACE OBJECTS ========================

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.dropout(x)
        return x


class FeedForward(nn.Module):
    def __init__(self, embed_dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio=4, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = FeedForward(embed_dim, int(embed_dim * mlp_ratio), dropout)
        
    def forward(self, x, mask=None):
        x = x + self.attn(self.norm1(x), mask)
        x = x + self.mlp(self.norm2(x))
        return x


class BERTTransformer(nn.Module):
    def __init__(self, vocab_size=30522, max_seq_len=128, embed_dim=512, 
                 num_heads=8, num_layers=8, num_classes=1, dropout=0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.token_embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_seq_len, embed_dim))
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        self._init_weights()
        
    def _init_weights(self):
        nn.init.normal_(self.token_embed.weight, std=0.02)
        nn.init.normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)
        
    def forward(self, input_ids, mask=None):
        B, N = input_ids.shape
        x = self.token_embed(input_ids)
        x = x + self.pos_embed[:, :N, :]
        x = self.dropout(x)
        for block in self.blocks:
            x = block(x, mask)
        x = self.norm(x)
        x = x[:, 0]
        x = self.head(x)
        return x


def create_bert_50m(num_classes=1):
    return BERTTransformer(
        vocab_size=30522, max_seq_len=128, embed_dim=768,
        num_heads=12, num_layers=8, num_classes=num_classes, dropout=0.05
    )


class YelpReviewDataset(Dataset):
    def __init__(self, split='train', max_length=128, subset_size=10000):
        raw_data = load_dataset("yelp_review_full", split=split)
        self.data = raw_data.select(range(min(len(raw_data), subset_size)))
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.max_length = max_length
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        encoding = self.tokenizer(
            item['text'],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        input_ids = encoding['input_ids'].squeeze(0)
        label = torch.tensor(item['label'] / 4.0, dtype=torch.float32)
        return input_ids, label


# ==================== BERT WEIGHT SPACE OBJECTS ====================

class BERTAttentionWeights:
    """Weight space object for BERT attention layers"""
    def __init__(self, qkv_weight, qkv_bias, proj_weight, proj_bias, num_heads):
        self.qkv_weight = qkv_weight
        self.qkv_bias = qkv_bias
        self.proj_weight = proj_weight
        self.proj_bias = proj_bias
        self.num_heads = num_heads
        
    def split_heads(self):
        """Split QKV weights into per-head Q, K, V matrices"""
        embed_dim = self.qkv_weight.shape[1]
        head_dim = embed_dim // self.num_heads
        
        # QKV weight is [3*embed_dim, embed_dim]
        qkv = self.qkv_weight.reshape(3, embed_dim, embed_dim)
        q_weight, k_weight, v_weight = qkv[0], qkv[1], qkv[2]
        
        # Split into heads: [num_heads, head_dim, embed_dim]
        q_heads = q_weight.reshape(self.num_heads, head_dim, embed_dim)
        k_heads = k_weight.reshape(self.num_heads, head_dim, embed_dim)
        v_heads = v_weight.reshape(self.num_heads, head_dim, embed_dim)
        
        return q_heads, k_heads, v_heads


class BERTTransformerBlockWeights:
    """
    Weight space object for a BERT transformer block
    NOW USES TUPLES like VisionTransformerWeightSpace!
    """
    def __init__(self, attention, norm1_weight, norm1_bias, 
                 mlp_weights, mlp_biases,
                 norm2_weight, norm2_bias):
        self.attention = attention
        self.norm1_weight = norm1_weight
        self.norm1_bias = norm1_bias
        self.mlp_weights = mlp_weights
        self.mlp_biases = mlp_biases
        self.norm2_weight = norm2_weight
        self.norm2_bias = norm2_bias


class BERTWeightSpace:
    """Weight space object for BERT Transformers"""
    
    def __init__(self, 
                 token_embed: torch.Tensor,
                 pos_embed: torch.Tensor,
                 blocks: List[BERTTransformerBlockWeights],
                 norm_weight: torch.Tensor,
                 norm_bias: torch.Tensor,
                 head_weight: torch.Tensor,
                 head_bias: torch.Tensor):
        
        self.token_embed = token_embed
        self.pos_embed = pos_embed
        self.blocks = blocks
        self.norm_weight = norm_weight
        self.norm_bias = norm_bias
        self.head_weight = head_weight
        self.head_bias = head_bias
        
    @classmethod
    def from_bert_model(cls, model: nn.Module):
        """Extract weights from a BERT model"""
        blocks = []
        
        for block in model.blocks:
            attn = block.attn
            attention_weights = BERTAttentionWeights(
                qkv_weight=attn.qkv.weight.data.clone(),
                qkv_bias=attn.qkv.bias.data.clone() if attn.qkv.bias is not None else None,
                proj_weight=attn.proj.weight.data.clone(),
                proj_bias=attn.proj.bias.data.clone() if attn.proj.bias is not None else None,
                num_heads=attn.num_heads
            )
            
            mlp_weights = []
            mlp_biases = []
            for name, layer in block.mlp.named_children():
                if hasattr(layer, 'weight'):
                    mlp_weights.append(layer.weight.data.clone())
                    if hasattr(layer, 'bias') and layer.bias is not None:
                        mlp_biases.append(layer.bias.data.clone())
            mlp_weights = tuple(mlp_weights)
            mlp_biases = tuple(mlp_biases)
            
            block_weights = BERTTransformerBlockWeights(
                attention=attention_weights,
                norm1_weight=block.norm1.weight.data.clone(),
                norm1_bias=block.norm1.bias.data.clone(),
                mlp_weights=mlp_weights,
                mlp_biases=mlp_biases,
                norm2_weight=block.norm2.weight.data.clone(),
                norm2_bias=block.norm2.bias.data.clone()
            )
            blocks.append(block_weights)
        
        return cls(
            token_embed=model.token_embed.weight.data.clone(),
            pos_embed=model.pos_embed.data.clone(),
            blocks=blocks,
            norm_weight=model.norm.weight.data.clone(),
            norm_bias=model.norm.bias.data.clone(),
            head_weight=model.head.weight.data.clone(),
            head_bias=model.head.bias.data.clone()
        )
    
    def apply_to_model(self, model: nn.Module):
        """Apply weights to a BERT model"""
        with torch.no_grad():
            model.token_embed.weight.data.copy_(self.token_embed)
            model.pos_embed.data.copy_(self.pos_embed)
            
            for block, block_weights in zip(model.blocks, self.blocks):
                attn = block.attn
                attn.qkv.weight.data.copy_(block_weights.attention.qkv_weight)
                if block_weights.attention.qkv_bias is not None:
                    attn.qkv.bias.data.copy_(block_weights.attention.qkv_bias)
                attn.proj.weight.data.copy_(block_weights.attention.proj_weight)
                if block_weights.attention.proj_bias is not None:
                    attn.proj.bias.data.copy_(block_weights.attention.proj_bias)
                
                block.norm1.weight.data.copy_(block_weights.norm1_weight)
                block.norm1.bias.data.copy_(block_weights.norm1_bias)
                block.norm2.weight.data.copy_(block_weights.norm2_weight)
                block.norm2.bias.data.copy_(block_weights.norm2_bias)
                
                mlp_layers = [layer for name, layer in block.mlp.named_children() 
                             if hasattr(layer, 'weight')]
                for layer, weight in zip(mlp_layers, block_weights.mlp_weights):
                    layer.weight.data.copy_(weight)
                    
                mlp_bias_idx = 0
                for name, layer in block.mlp.named_children():
                    if hasattr(layer, 'bias') and layer.bias is not None:
                        if mlp_bias_idx < len(block_weights.mlp_biases):
                            layer.bias.data.copy_(block_weights.mlp_biases[mlp_bias_idx])
                            mlp_bias_idx += 1
            
            model.norm.weight.data.copy_(self.norm_weight)
            model.norm.bias.data.copy_(self.norm_bias)
            model.head.weight.data.copy_(self.head_weight)
            model.head.bias.data.copy_(self.head_bias)
    
    def flatten(self, device=None) -> torch.Tensor:
        """Flatten all weights into a single vector - GPU-safe"""
        all_params = []
        
        all_params.append(self.token_embed.cpu().flatten())
        all_params.append(self.pos_embed.cpu().flatten())
        
        for block in self.blocks:
            all_params.append(block.attention.qkv_weight.cpu().flatten())
            if block.attention.qkv_bias is not None:
                all_params.append(block.attention.qkv_bias.cpu().flatten())
            all_params.append(block.attention.proj_weight.cpu().flatten())
            if block.attention.proj_bias is not None:
                all_params.append(block.attention.proj_bias.cpu().flatten())
            
            all_params.append(block.norm1_weight.cpu().flatten())
            all_params.append(block.norm1_bias.cpu().flatten())
            all_params.append(block.norm2_weight.cpu().flatten())
            all_params.append(block.norm2_bias.cpu().flatten())
            
            for w in block.mlp_weights:
                all_params.append(w.cpu().flatten())
            for b in block.mlp_biases:
                all_params.append(b.cpu().flatten())
        
        all_params.append(self.norm_weight.cpu().flatten())
        all_params.append(self.norm_bias.cpu().flatten())
        all_params.append(self.head_weight.cpu().flatten())
        all_params.append(self.head_bias.cpu().flatten())
        
        # Concatenate on CPU (safe!)
        flat = torch.cat(all_params)
        
        if device and device != 'cpu':
            flat = flat.to(device)
        
        return flat
    
    @classmethod
    def from_flat(cls, flat_tensor, reference_ws, device=None):
        """Reconstruct BERTWeightSpace from flattened weights"""
        if device is None:
            device = flat_tensor.device
            
        param_shapes = []
        param_types = []
        
        param_shapes.append(reference_ws.token_embed.shape)
        param_types.append('token_embed')
        
        param_shapes.append(reference_ws.pos_embed.shape)
        param_types.append('pos_embed')
        
        for block_idx, block in enumerate(reference_ws.blocks):
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
            
            param_shapes.append(block.norm1_weight.shape)
            param_types.append(f'block_{block_idx}_norm1_weight')
            
            param_shapes.append(block.norm1_bias.shape)
            param_types.append(f'block_{block_idx}_norm1_bias')
            
            param_shapes.append(block.norm2_weight.shape)
            param_types.append(f'block_{block_idx}_norm2_weight')
            
            param_shapes.append(block.norm2_bias.shape)
            param_types.append(f'block_{block_idx}_norm2_bias')
            
            for mlp_idx, mlp_weight in enumerate(block.mlp_weights):
                param_shapes.append(mlp_weight.shape)
                param_types.append(f'block_{block_idx}_mlp_weight_{mlp_idx}')
            
            for mlp_idx, mlp_bias in enumerate(block.mlp_biases):
                param_shapes.append(mlp_bias.shape)
                param_types.append(f'block_{block_idx}_mlp_bias_{mlp_idx}')
        
        param_shapes.append(reference_ws.norm_weight.shape)
        param_types.append('norm_weight')
        
        param_shapes.append(reference_ws.norm_bias.shape)
        param_types.append('norm_bias')
        
        param_shapes.append(reference_ws.head_weight.shape)
        param_types.append('head_weight')
        
        param_shapes.append(reference_ws.head_bias.shape)
        param_types.append('head_bias')
        
        sizes = [np.prod(shape) for shape in param_shapes]
        parts = []
        start = 0
        
        for size in sizes:
            parts.append(flat_tensor[start:start+size])
            start += size
        
        reconstructed_params = {}
        for i, (shape, param_type) in enumerate(zip(param_shapes, param_types)):
            reconstructed_params[param_type] = parts[i].reshape(shape).to(device)
        
        reconstructed_blocks = []
        num_blocks = len(reference_ws.blocks)
        
        for block_idx in range(num_blocks):
            qkv_weight = reconstructed_params[f'block_{block_idx}_attn_qkv_weight']
            qkv_bias = reconstructed_params.get(f'block_{block_idx}_attn_qkv_bias', None)
            proj_weight = reconstructed_params[f'block_{block_idx}_attn_proj_weight']  
            proj_bias = reconstructed_params.get(f'block_{block_idx}_attn_proj_bias', None)
            
            attention = BERTAttentionWeights(
                qkv_weight=qkv_weight,
                qkv_bias=qkv_bias,
                proj_weight=proj_weight,
                proj_bias=proj_bias,
                num_heads=reference_ws.blocks[block_idx].attention.num_heads
            )
            
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
            
            block = BERTTransformerBlockWeights(
                attention=attention,
                norm1_weight=reconstructed_params[f'block_{block_idx}_norm1_weight'],
                norm1_bias=reconstructed_params[f'block_{block_idx}_norm1_bias'],
                mlp_weights=tuple(mlp_weights),
                mlp_biases=tuple(mlp_biases),
                norm2_weight=reconstructed_params[f'block_{block_idx}_norm2_weight'],
                norm2_bias=reconstructed_params[f'block_{block_idx}_norm2_bias']
            )
            
            reconstructed_blocks.append(block)
        
        return cls(
            token_embed=reconstructed_params['token_embed'],
            pos_embed=reconstructed_params['pos_embed'],
            blocks=reconstructed_blocks,
            norm_weight=reconstructed_params['norm_weight'], 
            norm_bias=reconstructed_params['norm_bias'],
            head_weight=reconstructed_params['head_weight'],
            head_bias=reconstructed_params['head_bias']
        )


class BERTPermutationSpec:
    """Specification for BERT permutations"""
    def __init__(self, num_blocks: int):
        self.num_blocks = num_blocks
        self.block_perms = []
        for _ in range(num_blocks):
            self.block_perms.append({
                'attention_out': None,
                'mlp1': None,
            })
    
    def set_block_perm(self, block_idx: int, perm_type: str, perm: torch.Tensor):
        """Set a specific permutation for a block"""
        self.block_perms[block_idx][perm_type] = perm


class TransFusionMatcher:
    """
    Weight matching using TransFusion approach for Vision Transformers
    """
    
    def __init__(self, num_iterations: int = 5, epsilon: float = 1e-8):
        self.num_iterations = num_iterations
        self.epsilon = epsilon
        
    def compute_spectral_distance(self, weight1: torch.Tensor, weight2: torch.Tensor) -> float:
        """Compute permutation-invariant distance using singular values"""
        try:
            _, s1, _ = torch.svd(weight1)
            _, s2, _ = torch.svd(weight2)
        except:
            _, s1, _ = np.linalg.svd(weight1.cpu().numpy())
            _, s2, _ = np.linalg.svd(weight2.cpu().numpy())
            s1 = torch.tensor(s1)
            s2 = torch.tensor(s2)
        
        max_len = max(len(s1), len(s2))
        if len(s1) < max_len:
            s1 = torch.cat([s1, torch.zeros(max_len - len(s1))])
        if len(s2) < max_len:
            s2 = torch.cat([s2, torch.zeros(max_len - len(s2))])
        
        return torch.norm(s1 - s2).item()
    
    def compose_attention_permutation(self, inter_head_perm: torch.Tensor,
                                     intra_head_perms: List[torch.Tensor],
                                     d_model: int, num_heads: int) -> torch.Tensor:
        """Compose inter and intra head permutations into a single block diagonal matrix"""
        head_dim = d_model // num_heads
        P_attn = torch.zeros(d_model, d_model)
        
        for i in range(num_heads):
            j = torch.argmax(inter_head_perm[:, i]).item()
            P_intra = intra_head_perms[j] if j < len(intra_head_perms) else torch.eye(head_dim)
            
            start_i = i * head_dim
            end_i = (i + 1) * head_dim
            start_j = j * head_dim
            end_j = (j + 1) * head_dim
            
            P_attn[start_i:end_i, start_j:end_j] = P_intra
        
        return P_attn
    
    def match_attention_heads(self, attn1, attn2) -> Tuple[torch.Tensor, List[torch.Tensor], torch.Tensor]:
        """Two-level matching for attention heads"""
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
    
    def match_mlp_layer(self, weight1: torch.Tensor, weight2: torch.Tensor,
                       prev_perm: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Match MLP layers using Hungarian algorithm"""
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
    
    def apply_permutation_to_weights(self, weights, perm_spec) -> Any:
        """Apply computed permutations to weight space"""
        result = copy.deepcopy(weights)
        
        prev_output_perm = None
        
        for block_idx, (source_block, result_block) in enumerate(zip(weights.blocks, result.blocks)):
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
                
                result_block.mlp_weights[1] = torch.mm(result_block.mlp_weights[1], P_mlp1.t())
                
                result_block.mlp_weights = tuple(result_block.mlp_weights)
                result_block.mlp_biases = tuple(result_block.mlp_biases)
                
                prev_output_perm = torch.eye(d_model, device=device)
            else:
                prev_output_perm = torch.eye(d_model, device=device)
        
        return result
    
    def canonicalize_model(self, models: List[Any], reference_idx: int = 0) -> List[Any]:
        """Canonicalize multiple models using one as reference"""
        reference = models[reference_idx]
        canonicalized = []
        
        for i, model in enumerate(models):
            if i == reference_idx:
                canonicalized.append(reference)
            else:
                perm_spec = BERTPermutationSpec(len(model.blocks))
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
                    
                    current_model = self.apply_permutation_to_weights(current_model, perm_spec)

                
                canonicalized.append(current_model)
        
        return canonicalized


# ==================== UTILITY FUNCTIONS ====================

def evaluate_model(model, loader, device):
    """Evaluate model and return R² score"""
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for input_ids, y in loader:
            input_ids, y = input_ids.to(device), y.to(device)
            out = model(input_ids).squeeze(-1)
            preds.extend(out.cpu().numpy())
            labels.extend(y.cpu().numpy())
    r2 = r2_score(labels, preds)
    return r2


# ==================== MAIN TRANSFUSION FUNCTION ===================

class FlowMatching:
    def __init__(self, sourceloader, targetloader, model, mode="velocity", 
                 t_dist="uniform", device=None, normalize_pred=False, geometric=False):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.sourceloader = sourceloader
        self.targetloader = targetloader
        self.model = model.to(self.device)
        self.mode = mode
        self.t_dist = t_dist
        self.sigma = 0.001
        self.normalize_pred = normalize_pred
        self.geometric = geometric
        
        # Initialize iterators properly
        self.source_iter = None
        self.target_iter = None
        
        self.metrics = {"train_loss": [], "time": [], "grad_norm": [], "flow_norm": [], "true_norm": []}
        self.best_loss = float('inf')
        self.best_model_state = None
        self.input_dim = None

    def sample_from_loader(self, loader, is_source=True):
        """ROBUST sampling method with proper error handling"""
        try:
            iterator_attr = 'source_iter' if is_source else 'target_iter'
            
            if getattr(self, iterator_attr) is None:
                setattr(self, iterator_attr, iter(loader))
            
            try:
                batch = next(getattr(self, iterator_attr))
            except StopIteration:
                setattr(self, iterator_attr, iter(loader))
                batch = next(getattr(self, iterator_attr))
            
            if isinstance(batch, (list, tuple)):
                data = batch[0]
            else:
                data = batch
                
            return data.to(self.device)
            
        except Exception as e:
            logging.warning(f"Error sampling from loader: {str(e)}")
            try:
                sample = loader.dataset[0]
                if isinstance(sample, (list, tuple)):
                    sample_data = sample[0]
                else:
                    sample_data = sample
                shape = sample_data.shape
                return torch.zeros(loader.batch_size, *shape, device=self.device)
            except:
                return torch.zeros(loader.batch_size, 1, device=self.device)

    def sample_time_and_flow(self):
        """Sample time t and flow with proper memory management"""
        x0 = self.sample_from_loader(self.sourceloader, is_source=True)
        x1 = self.sample_from_loader(self.targetloader, is_source=False)
        
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

    # ========== ADD THESE MISSING METHODS ==========
    
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

    def train(self, n_iters=10, optimizer=None, scheduler=None, sigma=0.001, patience=1e99, log_freq=5, accum_steps=4):
        """Train the flow model with gradient accumulation"""
        self.sigma = sigma
        self.metrics = {"train_loss": [], "time": [], "grad_norm": [], "flow_norm": [], "true_norm": []}
        last_loss = 1e99
        patience_count = 0
    
        pbar = tqdm(range(n_iters), desc="Training steps")
        optimizer.zero_grad()
        
        for i in pbar:
            try:
                flow = self.sample_time_and_flow()
    
                _, flow_pred = self.forward(flow)
                _, loss = self.loss_fn(flow_pred, flow)
    
                if not torch.isnan(loss) and not torch.isinf(loss):
                    (loss / accum_steps).backward()
    
                    if (i + 1) % accum_steps == 0:
                        optimizer.step()
                        optimizer.zero_grad()
    
                        if scheduler is not None:
                            scheduler.step()
    
                    if loss.item() < self.best_loss:
                        self.best_loss = loss.item()
                        self.best_model_state = {k: v.clone() for k, v in self.model.state_dict().items()}
                else:
                    logging.info(f"Skipping step {i} due to invalid loss: {loss.item()}")
                    continue
    
                if loss.item() > last_loss:
                    patience_count += 1
                    if patience_count >= patience:
                        logging.info(f"Early stopping at iteration {i}")
                        break
                else:
                    patience_count = 0
    
                last_loss = loss.item()
    
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


def apply_transfusion(model_dir, num_models, pretrained_model_name, ref_point=0, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load reference model
    ref_model = create_bert_50m(num_classes=1).to(device)
    ref_model_path = f"{model_dir}/{pretrained_model_name}{ref_point}_best.pt"
    
    try:
        ref_model.load_state_dict(torch.load(ref_model_path, map_location=device))
        logging.info(f"Loaded reference model from {ref_model_path}")
    except Exception as e:
        logging.error(f"Failed to load reference model: {e}")
        raise e
    
    # Extract reference weight space and move to CPU
    ref_ws = BERTWeightSpace.from_bert_model(ref_model)
    ref_ws_cpu = move_weight_space_to_cpu(ref_ws)
    
    weight_space_objects = [ref_ws_cpu]
    
    # Aggressive cleanup
    del ref_model, ref_ws
    aggressive_cleanup()
    
    matcher = TransFusionMatcher(num_iterations=2)
    
    for i in tqdm(range(num_models), desc="Processing BERT models"):
        if i == ref_point:
            continue
        
        model_path = f"{model_dir}/{pretrained_model_name}{i}_best.pt"
        if not os.path.exists(model_path):
            logging.warning(f"Skipping model {i} - file not found")
            continue
        
        try:
            # Load model on GPU
            model = create_bert_50m(num_classes=1).to(device)
            model.load_state_dict(torch.load(model_path, map_location=device))
            
            # Extract weight space
            ws = BERTWeightSpace.from_bert_model(model)
            ws_cpu = move_weight_space_to_cpu(ws)
            
            # Delete GPU version immediately
            del model, ws
            aggressive_cleanup()
            
            # Canonicalize (temporarily move to GPU)
            ref_ws_gpu = move_weight_space_to_device(ref_ws_cpu, device)
            ws_gpu = move_weight_space_to_device(ws_cpu, device)
            
            canonicalized_list = matcher.canonicalize_model([ref_ws_gpu, ws_gpu], reference_idx=0)
            aligned_ws = canonicalized_list[1]
            
            # Move back to CPU and delete GPU versions
            aligned_ws_cpu = move_weight_space_to_cpu(aligned_ws)
            weight_space_objects.append(aligned_ws_cpu)
            
            del ws_cpu, ref_ws_gpu, ws_gpu, canonicalized_list, aligned_ws
            aggressive_cleanup()
            
            if (i + 1) % 5 == 0:  # More frequent cleanup
                logging.info(f"Processed {i+1}/{num_models} - Memory cleared")
                
        except Exception as e:
            logging.error(f"Error processing model {i}: {str(e)}")
            aggressive_cleanup()
            continue
    
    logging.info(f"Successfully processed {len(weight_space_objects)} BERT models")
    aggressive_cleanup()
    
    return ref_ws_cpu, weight_space_objects

def aggressive_cleanup():
    """Force garbage collection and GPU memory cleanup"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

def move_weight_space_to_cpu(ws):
    """
    Move to CPU without deep copy when possible
    """
    # Create new weight space with CPU tensors (no deepcopy)
    blocks_cpu = []
    for block in ws.blocks:
        # Create new block objects but share structure
        attn_cpu = BERTAttentionWeights(
            qkv_weight=block.attention.qkv_weight.cpu(),
            qkv_bias=block.attention.qkv_bias.cpu() if block.attention.qkv_bias is not None else None,
            proj_weight=block.attention.proj_weight.cpu(),
            proj_bias=block.attention.proj_bias.cpu() if block.attention.proj_bias is not None else None,
            num_heads=block.attention.num_heads
        )
        
        block_cpu = BERTTransformerBlockWeights(
            attention=attn_cpu,
            norm1_weight=block.norm1_weight.cpu(),
            norm1_bias=block.norm1_bias.cpu(),
            mlp_weights=tuple(w.cpu() for w in block.mlp_weights),
            mlp_biases=tuple(b.cpu() for b in block.mlp_biases),
            norm2_weight=block.norm2_weight.cpu(),
            norm2_bias=block.norm2_bias.cpu()
        )
        blocks_cpu.append(block_cpu)
    
    return BERTWeightSpace(
        token_embed=ws.token_embed.cpu(),
        pos_embed=ws.pos_embed.cpu(),
        blocks=blocks_cpu,
        norm_weight=ws.norm_weight.cpu(),
        norm_bias=ws.norm_bias.cpu(),
        head_weight=ws.head_weight.cpu(),
        head_bias=ws.head_bias.cpu()
    )


def move_weight_space_to_device(ws_cpu, device):
    # Create new blocks without deepcopy
    blocks_dev = []
    for block_cpu in ws_cpu.blocks:
        attn_dev = BERTAttentionWeights(
            qkv_weight=block_cpu.attention.qkv_weight.to(device),
            qkv_bias=block_cpu.attention.qkv_bias.to(device) if block_cpu.attention.qkv_bias is not None else None,
            proj_weight=block_cpu.attention.proj_weight.to(device),
            proj_bias=block_cpu.attention.proj_bias.to(device) if block_cpu.attention.proj_bias is not None else None,
            num_heads=block_cpu.attention.num_heads
        )
        
        block_dev = BERTTransformerBlockWeights(
            attention=attn_dev,
            norm1_weight=block_cpu.norm1_weight.to(device),
            norm1_bias=block_cpu.norm1_bias.to(device),
            mlp_weights=tuple(w.to(device) for w in block_cpu.mlp_weights),
            mlp_biases=tuple(b.to(device) for b in block_cpu.mlp_biases),
            norm2_weight=block_cpu.norm2_weight.to(device),
            norm2_bias=block_cpu.norm2_bias.to(device)
        )
        blocks_dev.append(block_dev)
    
    return BERTWeightSpace(
        token_embed=ws_cpu.token_embed.to(device),
        pos_embed=ws_cpu.pos_embed.to(device),
        blocks=blocks_dev,
        norm_weight=ws_cpu.norm_weight.to(device),
        norm_bias=ws_cpu.norm_bias.to(device),
        head_weight=ws_cpu.head_weight.to(device),
        head_bias=ws_cpu.head_bias.to(device)
    )

def flatten_weights_in_batches(weight_space_objects, batch_size=10, device='cpu'):
    """Process weights in batches to avoid OOM"""
    all_flat_weights = []
    num_batches = (len(weight_space_objects) + batch_size - 1) // batch_size
    
    logging.info(f"Flattening {len(weight_space_objects)} weight spaces in {num_batches} batches...")
    
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(weight_space_objects))
        batch = weight_space_objects[start_idx:end_idx]
        
        batch_flat = []
        for ws in batch:
            try:
                flat = ws.flatten(device=device)
                batch_flat.append(flat)
                del flat
            except Exception as e:
                logging.error(f"Error flattening weight space: {e}")
                continue
        
        all_flat_weights.extend(batch_flat)
        
        # Clean up after each batch
        del batch, batch_flat
        torch.cuda.empty_cache()
        gc.collect()
        
        if (batch_idx + 1) % 5 == 0:
            logging.info(f"Processed batch {batch_idx + 1}/{num_batches}")
    
    return all_flat_weights


class VisionTransformerFlowModel(nn.Module):
    """Flow model for ViT weight spaces"""
    
    def __init__(self, input_dim, time_embed_dim=64):
        super().__init__()
        self.input_dim = input_dim
        self.time_embed_dim = time_embed_dim
        
        self.time_embed = nn.Sequential(
            nn.Linear(1, time_embed_dim),
            nn.GELU(),
            nn.Linear(time_embed_dim, time_embed_dim)
        )
        
        hidden_dim = 1024
        logging.info(f"hidden_dim:{hidden_dim}")
        
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



import numpy as np
import logging
from tqdm import tqdm
from sklearn.utils.extmath import randomized_svd
from typing import Optional, Tuple, Callable
import tempfile

class BatchedCovariancePCA:
    """
    Incremental covariance-based PCA for maximum components.
    Handles n_components up to n_models (e.g., 99 or 100 components from 100 models).
    """
    
    def __init__(self, n_components: int, seed: int = 42):
        self.n_components = n_components
        self.seed = seed
        self.mean_ = None
        self.components_ = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None
        self.n_models_ = None
        self.n_params_ = None
        
    def fit(self, model_loader_func: Callable, n_models: int, batch_size: int = 20):
        """
        Fit PCA by building covariance matrix incrementally.
        
        Args:
            model_loader_func: Callable(start_idx, end_idx) -> array (n_params, n_models_in_batch)
            n_models: Total number of models
            batch_size: Models to load at once
        """
        logging.info(f"Building covariance matrix for {n_models} models...")
        logging.info(f"Batch size: {batch_size} models at a time")
        
        self.n_models_ = n_models
        
        # Pass 1: Compute mean incrementally
        logging.info("Pass 1: Computing mean...")
        mean_accumulator = None
        
        for start_idx in tqdm(range(0, n_models, batch_size), desc="Mean computation"):
            end_idx = min(start_idx + batch_size, n_models)
            batch = model_loader_func(start_idx, end_idx)
            
            if mean_accumulator is None:
                self.n_params_ = batch.shape[0]
                mean_accumulator = np.zeros(self.n_params_, dtype=np.float64)
            
            mean_accumulator += np.sum(batch, axis=1, dtype=np.float64)
            del batch
            gc.collect()
        
        self.mean_ = mean_accumulator / n_models
        del mean_accumulator
        gc.collect()
        
        # Pass 2: Build covariance matrix (n_models × n_models) - STREAMING
        logging.info(f"Pass 2: Building covariance matrix ({n_models}×{n_models}) - streaming...")
        C = np.zeros((n_models, n_models), dtype=np.float64)
        
        for i_start in tqdm(range(0, n_models, batch_size), desc="Covariance rows"):
            i_end = min(i_start + batch_size, n_models)
            batch_i = model_loader_func(i_start, i_end)
            batch_i_centered = batch_i - self.mean_[:, np.newaxis]
            
            for j_start in range(0, n_models, batch_size):
                j_end = min(j_start + batch_size, n_models)
                batch_j = model_loader_func(j_start, j_end)
                batch_j_centered = batch_j - self.mean_[:, np.newaxis]
                
                C[i_start:i_end, j_start:j_end] = batch_i_centered.T @ batch_j_centered
                
                del batch_j, batch_j_centered
                gc.collect()
            
            del batch_i, batch_i_centered
            gc.collect()
        
        # Pass 3: SVD on covariance matrix
        logging.info(f"Pass 3: SVD on covariance matrix ({n_models}×{n_models})...")
        n_components = min(self.n_components, n_models - 1)
        
        U, S, _ = randomized_svd(
            C, 
            n_components=n_components,
            n_iter=5,
            random_state=self.seed
        )
        
        eigenvalues = S / (n_models - 1)
        del C
        gc.collect()
        
        # Pass 4: Compute components in original space
        logging.info("Computing principal components in original space...")
        components = np.zeros((n_components, self.n_params_), dtype=np.float64)
        
        for start_idx in tqdm(range(0, n_models, batch_size), desc="Computing components"):
            end_idx = min(start_idx + batch_size, n_models)
            batch = model_loader_func(start_idx, end_idx)
            batch_centered = batch - self.mean_[:, np.newaxis]
            
            U_batch = U[start_idx:end_idx, :]
            
            for k in range(n_components):
                components[k, :] += batch_centered @ U_batch[:, k]
            
            del batch, batch_centered
            gc.collect()
        
        # Normalize components to unit length
        for k in range(n_components):
            norm = np.linalg.norm(components[k, :])
            if norm > 1e-10:
                components[k, :] /= norm
        
        self.components_ = components
        self.explained_variance_ = eigenvalues[:n_components]
        total_variance = np.sum(eigenvalues)
        self.explained_variance_ratio_ = eigenvalues[:n_components] / total_variance
        
        logging.info(f"✓ Fitted PCA with {n_components} components")
        logging.info(f"✓ Cumulative variance: {np.sum(self.explained_variance_ratio_):.2%}")
        
        return self
    
    def transform(self, model_loader_func: Callable, n_models: int, 
                  batch_size: int = 20, output_file: Optional[str] = None) -> str:
        """
        Project models to PCA space.
        
        Returns:
            Path to memmap file with projected data (n_models, n_components)
        """
        if output_file is None:
            output_file = tempfile.NamedTemporaryFile(delete=False, suffix=".npy").name
        
        projected = np.memmap(
            output_file, 
            dtype=np.float32, 
            mode="w+", 
            shape=(n_models, self.n_components)
        )
        
        logging.info(f"Transforming {n_models} models to {self.n_components}-dimensional space...")
        
        idx = 0
        for start_idx in tqdm(range(0, n_models, batch_size), desc="Transforming"):
            end_idx = min(start_idx + batch_size, n_models)
            batch = model_loader_func(start_idx, end_idx)  # (n_params, n_models_in_batch)
            batch_centered = batch - self.mean_[:, np.newaxis]
            
            # Project: (n_models_in_batch, n_components) = (n_models_in_batch, n_params) @ (n_params, n_components).T
            batch_projected = batch_centered.T @ self.components_.T
            
            projected[idx:idx + batch_projected.shape[0]] = batch_projected
            idx += batch_projected.shape[0]
            
            del batch, batch_centered, batch_projected
        
        projected.flush()
        logging.info(f"✓ Projected data saved to {output_file}")
        
        return output_file
    
    def inverse_transform(self, latent_vectors: np.ndarray) -> np.ndarray:
        """Reconstruct original models from PCA coordinates."""
        # X_reconstructed = latent @ components + mean
        return latent_vectors @ self.components_ + self.mean_


def flatten_and_project_to_disk_maxcomponents(
    model_loader_func: Callable,
    n_models: int = 100,
    n_components: Optional[int] = None,  # None = use maximum (n_models - 1)
    target_variance: Optional[float] = None,  # e.g., 0.99 for 99% variance
    batch_size: int = 20,
    output_file: Optional[str] = None
) -> Tuple[str, BatchedCovariancePCA, dict]:
    """
    PCA with maximum possible components (up to n_models).
    
    Args:
        model_loader_func: Callable(start_idx, end_idx) -> array (n_params, n_models_in_batch)
        n_models: Total number of models (e.g., 100)
        n_components: Number of components (None = auto-determine)
        target_variance: If set, auto-determine components for this variance (e.g., 0.99)
        batch_size: Models to load at once (e.g., 20)
        
    Returns:
        output_file: Path to projected data
        pca: Fitted PCA object
        info: Dictionary with variance info
    """
    # First fit with maximum components
    max_components = n_models - 1  # Maximum mathematically possible
    
    if n_components is None and target_variance is None:
        # Use maximum
        n_components = max_components
        logging.info(f"Using maximum components: {n_components}")
    elif target_variance is not None:
        # First fit with max to determine optimal number
        logging.info("Fitting with maximum components to determine optimal number...")
        pca_temp = BatchedCovariancePCA(n_components=max_components)
        pca_temp.fit(model_loader_func, n_models, batch_size)
        
        cumulative_variance = np.cumsum(pca_temp.explained_variance_ratio_)
        n_components = np.searchsorted(cumulative_variance, target_variance) + 1
        n_components = min(n_components, max_components)
        
        logging.info(f"Auto-selected {n_components} components for {target_variance:.1%} variance")
        logging.info(f"Actual variance: {cumulative_variance[n_components-1]:.2%}")
    
    # Fit with determined number of components
    pca = BatchedCovariancePCA(n_components=n_components)
    pca.fit(model_loader_func, n_models, batch_size)
    
    # Transform
    output_file = pca.transform(model_loader_func, n_models, batch_size, output_file)
    
    # Info
    info = {
        'n_components': n_components,
        'explained_variance': pca.explained_variance_,
        'explained_variance_ratio': pca.explained_variance_ratio_,
        'cumulative_variance': np.cumsum(pca.explained_variance_ratio_),
        'total_variance_captured': np.sum(pca.explained_variance_ratio_)
    }
    
    logging.info("=" * 50)
    logging.info(f"✓ COMPLETE!")
    logging.info(f"✓ Components: {n_components}/{n_models}")
    logging.info(f"✓ Variance captured: {info['total_variance_captured']:.4%}")
    logging.info(f"✓ Output shape: ({n_models}, {n_components})")
    logging.info(f"✓ Output file: {output_file}")
    logging.info("=" * 50)
    
    return output_file, pca, info

# ============================================================================
# ADD THESE DATASET CLASSES
# ============================================================================

class MemoryMappedLatentDataset(Dataset):
    """
    Dataset that reads from memory-mapped file (for target data).
    
    Efficiently loads PCA-projected latent codes from disk without loading
    entire dataset into memory. Uses numpy's memory mapping for fast access.
    
    Args:
        mmap_file: Path to memmap file created by PCA.transform()
        shape: Tuple of (n_samples, n_features) - REQUIRED for raw memmap files
        device: Device to load tensors to ('cpu' or 'cuda')
        dtype: Data type (default: np.float32)
        preload: If True, load all data to RAM (faster but uses more memory)
    """
    
    def __init__(self, mmap_file, shape, device='cpu', dtype=np.float32, preload=False):
        if not os.path.exists(mmap_file):
            raise FileNotFoundError(f"Memmap file not found: {mmap_file}")
        
        self.mmap_file = mmap_file
        self.device = device
        self.preload = preload
        
        # Load with memory mapping (raw binary file, not .npy)
        self.data = np.memmap(
            mmap_file,
            dtype=dtype,
            mode='r',
            shape=shape
        )
        
        # Validate shape
        if self.data.ndim != 2:
            raise ValueError(
                f"Expected 2D array (n_samples, n_features), got shape {self.data.shape}"
            )
        
        self.n_samples, self.n_features = self.data.shape
        
        # Optional: preload to RAM for faster access (if dataset is small)
        if preload:
            print(f"Preloading {self.n_samples} samples to RAM...")
            self.data = np.array(self.data, dtype=dtype)
            print("✓ Preload complete")
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        """
        Get a single sample or batch of samples.
        
        Args:
            idx: Integer index or slice
            
        Returns:
            Torch tensor on specified device
        """
        # Handle slicing for batch loading
        if isinstance(idx, slice):
            # Load slice from memmap (efficient!)
            sample = np.array(self.data[idx], dtype=np.float32)
        else:
            # Load single sample
            sample = np.array(self.data[idx], dtype=np.float32)
        
        return torch.from_numpy(sample).to(self.device)
    
    def get_batch(self, indices):
        """
        Get multiple samples by index list (more efficient than multiple __getitem__ calls).
        
        Args:
            indices: List or array of indices
            
        Returns:
            Torch tensor of shape (len(indices), n_features)
        """
        batch = np.array(self.data[indices], dtype=np.float32)
        return torch.from_numpy(batch).to(self.device)
    
    @property
    def shape(self):
        """Return dataset shape (n_samples, n_features)"""
        return self.data.shape
    
    def __repr__(self):
        return (f"MemoryMappedLatentDataset(n_samples={self.n_samples}, "
                f"n_features={self.n_features}, device='{self.device}', "
                f"dtype={self.data.dtype}, preload={self.preload})")


class StreamingLatentDataset(Dataset):
    """
    Generate noise on-the-fly (for source data).
    
    Creates random latent codes from Gaussian distribution without storing them.
    Memory-efficient for source distribution in flow matching.
    
    Args:
        n_samples: Number of samples in epoch
        latent_dim: Dimensionality of latent space
        std: Standard deviation of Gaussian noise
        device: Device to generate tensors on
        seed: Random seed for reproducibility (None = random each time)
    """
    
    def __init__(self, n_samples, latent_dim, std=0.001, device='cpu', seed=None):
        self.n_samples = n_samples
        self.latent_dim = latent_dim
        self.std = std
        self.device = device
        self.seed = seed
        
        # Optional: Set seed for reproducible "dataset"
        if seed is not None:
            self.generator = torch.Generator(device=device)
            self.generator.manual_seed(seed)
        else:
            self.generator = None
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        """
        Generate random sample on-the-fly.
        
        Note: idx is ignored (samples are iid), but needed for DataLoader compatibility.
        """
        if self.generator is not None:
            return torch.randn(
                self.latent_dim, 
                device=self.device, 
                generator=self.generator
            ) * self.std
        else:
            return torch.randn(self.latent_dim, device=self.device) * self.std
    
    def __repr__(self):
        return (f"StreamingLatentDataset(n_samples={self.n_samples}, "
                f"latent_dim={self.latent_dim}, std={self.std}, "
                f"device='{self.device}', seed={self.seed})")

def create_dataloaders_for_flow_matching(
    pca_output_file: str,
    n_models: int,
    latent_dim: int,
    source_std: float = 0.001,
    batch_size: int = 2,
    device: str = 'cpu',
    preload_target: bool = False
):
    """
    Create source and target dataloaders for flow matching.
    
    Args:
        pca_output_file: Path to PCA-projected memmap file
        n_models: Number of models in dataset
        latent_dim: Dimensionality of latent space
        source_std: Standard deviation for source distribution
        batch_size: Batch size for training
        device: Device to load data to
        preload_target: Whether to preload target data to RAM
        
    Returns:
        source_loader, target_loader
    """
    
    # Source: Random Gaussian noise (generated on-the-fly)
    source_dataset = StreamingLatentDataset(
        n_samples=n_models,
        latent_dim=latent_dim,
        std=source_std,
        device=device,
        seed=None  # Different noise each epoch
    )
    
    # Target: PCA-projected real models (memory-mapped)
    target_dataset = MemoryMappedLatentDataset(
        mmap_file=pca_output_file,
        shape=(n_models, latent_dim),  # ← REQUIRED for raw memmap files
        device=device,
        dtype=np.float32,
        preload=preload_target
    )
    
    # Verify dimensions match
    assert target_dataset.n_features == latent_dim, (
        f"Dimension mismatch: target has {target_dataset.n_features} features "
        f"but expected {latent_dim}"
    )
    
    # Create dataloaders
    source_loader = DataLoader(
        source_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(device == 'cuda')
    )
    
    target_loader = DataLoader(
        target_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # memmap doesn't pickle well with multiprocessing
        pin_memory=(device == 'cuda')
    )
    
    print(f"✓ Created dataloaders:")
    print(f"  Source: {len(source_dataset)} samples, {latent_dim}D")
    print(f"  Target: {len(target_dataset)} samples, {target_dataset.n_features}D")
    print(f"  Batch size: {batch_size}")
    
    return source_loader, target_loader
        
def train_flow_matching(model_dir="../yelp_bert_regression_models", 
                       num_models=20):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        ref_ws, weight_space_objects = apply_transfusion(
            model_dir=model_dir,
            num_models=num_models,
            pretrained_model_name="bert_",
            ref_point=0, 
            device=device
        )

        yelp_test_dataset = YelpReviewDataset('test', subset_size=10000)
        yelp_test_loader = DataLoader(yelp_test_dataset, batch_size=32)
 
        reference_ws = weight_space_objects[0]
        actual_num_models = len(weight_space_objects)
        
        def load_models_batch(start_idx, end_idx):
            """Load weight spaces as numpy array (n_params, n_models)"""
            batch = []
            for i in range(start_idx, end_idx):
                ws = weight_space_objects[i]
                flat = ws.flatten(device='cpu').numpy()
                batch.append(flat)
            return np.column_stack(batch)
    
        for init_type in ["gaussian_0.001"]:
            try:
                logging.info(f"Processing {init_type}")
                
                output_file, pca, info = flatten_and_project_to_disk_maxcomponents(
                    model_loader_func=load_models_batch,
                    n_models=actual_num_models,
                    target_variance=0.99,
                    batch_size=10
                )
                
                logging.info(f"pca info:{info}")
                
                latent_dim = info['n_components']
                logging.info(f"Latent dimension: {latent_dim}")
                                
                source_std = 0.01 if init_type == "gaussian_0.01" else 0.001
                
                source_dataset = StreamingLatentDataset(
                    n_samples=actual_num_models,
                    latent_dim=latent_dim,
                    std=source_std,
                    device='cpu'
                )

                target_dataset = MemoryMappedLatentDataset(
                    mmap_file=output_file,
                    shape=(actual_num_models, latent_dim),
                    device='cpu',
                    dtype=np.float32,
                    preload=False
                )

                flow_batch_size = 2
                sourceloader, targetloader = create_dataloaders_for_flow_matching(
                    pca_output_file=output_file,
                    n_models=actual_num_models,
                    latent_dim=latent_dim,
                    source_std=source_std,
                    batch_size=flow_batch_size,
                    device='cpu'
                )
                                
                logging.info(f"Source dataset: {len(source_dataset)} samples")
                logging.info(f"Target dataset: {len(target_dataset)} samples")
                

                # sourceloader = DataLoader(source_dataset, batch_size=flow_batch_size, shuffle=True)
                # targetloader = DataLoader(target_dataset, batch_size=flow_batch_size, shuffle=True)
                
                flow_model = VisionTransformerFlowModel(latent_dim).to(device)
                logging.info(f"Flow model parameters: {count_parameters(flow_model):,}")
                
                cfm = FlowMatching(
                    sourceloader=sourceloader,
                    targetloader=targetloader,
                    model=flow_model,
                    mode="velocity",
                    t_dist="beta",
                    device=device
                )
                
                cfm.input_dim = latent_dim 
                
                optimizer = torch.optim.AdamW(flow_model.parameters(), lr=1e-4, weight_decay=1e-5)
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30000, eta_min=1e-6)
                
                cfm.train(n_iters=30000, optimizer=optimizer, scheduler=scheduler, sigma=0.001, patience=100, log_freq=10)
                
                logging.info("Generating and evaluating new weights...")
                
                r2_scores = generate_and_evaluate_models(
                    cfm=cfm,
                    reference_ws=reference_ws,
                    pca=pca,
                    source_std=source_std,
                    device=device,
                    test_loader=yelp_test_loader,
                    n_samples=25,
                    gen_method="rk4",
                    batch_size=5
                )
                
                try:
                    os.remove(output_file)
                    logging.info(f"Cleaned up: {output_file}")
                except:
                    pass
                
                del cfm, flow_model, optimizer, scheduler
                aggressive_cleanup()
                
            except Exception as e:
                logging.error(f"Error in init_type {init_type}: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    except Exception as e:
        logging.error(f"Fatal error in train_flow_matching: {e}")
        import traceback
        traceback.print_exc()
        aggressive_cleanup()
        raise


def generate_and_evaluate_models(cfm, reference_ws, pca, source_std, 
                                 device, test_loader, n_samples=25, 
                                 gen_method="rk4", batch_size=5):
    """
    Generate models in batches with memory-efficient inverse transform.
    """
    logging.info(f"Generating {n_samples} models in batches of {batch_size}...")
    all_r2_scores = []
    
    for batch_start in range(0, n_samples, batch_size):
        batch_end = min(batch_start + batch_size, n_samples)
        batch_n = batch_end - batch_start
        
        logging.info(f"Generating models {batch_start} to {batch_end-1}...")
        
        try:
            # Generate latent codes on GPU
            random_latent = torch.randn(batch_n, cfm.model.input_dim, device=device) * source_std
            generated_latent_gpu = cfm.map(random_latent, n_steps=100, method=gen_method)
            
            # Move to CPU immediately
            generated_latent_cpu = generated_latent_gpu.cpu().numpy().astype(np.float32)
            del generated_latent_gpu, random_latent
            aggressive_cleanup()
            
            logging.info(f"  Inverse transforming {batch_n} samples...")
            generated_flat_cpu = pca.inverse_transform(generated_latent_cpu)
            del generated_latent_cpu
            gc.collect()
            
            for i in range(batch_n):
                try:
                    flat_gpu = torch.tensor(
                        generated_flat_cpu[i], 
                        dtype=torch.float32, 
                        device=device
                    )
                    
                    # Reconstruct weight space
                    generated_ws = BERTWeightSpace.from_flat(flat_gpu, reference_ws, device)
                    del flat_gpu
                    
                    # Create and evaluate model
                    new_model = create_bert_50m().to(device)
                    generated_ws.apply_to_model(new_model)
                    
                    r2 = evaluate_model(new_model, test_loader, device)
                    all_r2_scores.append(r2)
                    logging.info(f"  Model {batch_start + i} R²: {r2:.4f}")
                    
                    # Aggressive cleanup
                    del generated_ws, new_model
                    aggressive_cleanup()
                    
                except Exception as e:
                    logging.error(f"  Error generating model {batch_start + i}: {e}")
                    aggressive_cleanup()
                    continue
            
            # Clean up batch
            del generated_flat_cpu
            gc.collect()
            
        except Exception as e:
            logging.error(f"Error in batch {batch_start}-{batch_end}: {e}")
            aggressive_cleanup()
            continue
    
    # Print summary
    if all_r2_scores:
        logging.info("GENERATION RESULTS:")
        logging.info(f"  Generated: {len(all_r2_scores)}/{n_samples} models")
        logging.info(f"  Average R²: {np.mean(all_r2_scores):.4f} ± {np.std(all_r2_scores):.4f}")
        logging.info(f"  Min R²: {np.min(all_r2_scores):.4f}")
        logging.info(f"  Max R²: {np.max(all_r2_scores):.4f}")
    else:
        logging.error("No models were successfully generated!")
    
    return all_r2_scores


def main():
    """Main function demonstrating BERT flow matching"""
    logging.info("Starting BERT Flow Matching Pipeline...")
        
    train_flow_matching(
        model_dir="../yelp_bert_regression_models",
        num_models=100,
    )   

if __name__ == "__main__":
    logging.info("BERT Flow Matching")
    main()