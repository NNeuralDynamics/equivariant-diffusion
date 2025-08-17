import torch
import torch.nn as nn
import numpy as np
from scipy.optimize import linear_sum_assignment
from typing import List, Dict, Tuple, Optional
from .weight_space import VisionTransformerWeightSpace, AttentionWeights, TransformerBlockWeights
import copy


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
        # Compute SVD
        try:
            _, s1, _ = torch.svd(weight1)
            _, s2, _ = torch.svd(weight2)
        except:
            # Fallback to numpy if torch SVD fails
            _, s1, _ = np.linalg.svd(weight1.numpy())
            _, s2, _ = np.linalg.svd(weight2.numpy())
            s1 = torch.tensor(s1)
            s2 = torch.tensor(s2)
        
        # Pad to same length if necessary
        max_len = max(len(s1), len(s2))
        if len(s1) < max_len:
            s1 = torch.cat([s1, torch.zeros(max_len - len(s1))])
        if len(s2) < max_len:
            s2 = torch.cat([s2, torch.zeros(max_len - len(s2))])
        
        # Compute L2 distance between singular values
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
        
        # Step 1: Inter-head alignment
        distance_matrix = torch.zeros(num_heads, num_heads)
        
        for i in range(num_heads):
            for j in range(num_heads):
                # Sum distances for Q, K, V using spectral distance
                dist_q = self.compute_spectral_distance(q1[i], q2[j])
                dist_k = self.compute_spectral_distance(k1[i], k2[j])
                dist_v = self.compute_spectral_distance(v1[i], v2[j])
                distance_matrix[i, j] = dist_q + dist_k + dist_v
        
        # Solve assignment problem
        row_ind, col_ind = linear_sum_assignment(distance_matrix.numpy())
        inter_head_perm = torch.zeros(num_heads, num_heads)
        inter_head_perm[row_ind, col_ind] = 1.0
        
        # Step 2: Intra-head alignment
        intra_head_perms = []
        for i, j in zip(row_ind, col_ind):
            # Match individual units within paired heads
            # Create combined cost matrix for Q, K, V
            cost_q = -torch.mm(q2[j], q1[i].t())
            cost_k = -torch.mm(k2[j], k1[i].t())
            cost_v = -torch.mm(v2[j], v1[i].t())
            cost_matrix = (cost_q + cost_k + cost_v) / 3.0
            
            row_ind_intra, col_ind_intra = linear_sum_assignment(cost_matrix.numpy())
            
            perm = torch.zeros(head_dim, head_dim)
            perm[row_ind_intra, col_ind_intra] = 1.0
            intra_head_perms.append(perm)
        
        # Compose into single permutation
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
        # weight1 shape: [out_features, in_features]
        # weight2 shape: [out_features, in_features]
        
        # Apply previous layer's permutation if exists
        if prev_perm is not None:
            # prev_perm should match input dimension size
            if prev_perm.shape[0] == weight1.shape[1]:
                # Apply to input dimension: W' = W @ P^T
                weight1_permuted = torch.mm(weight1, prev_perm.t())
            else:
                # Size mismatch - skip permutation
                weight1_permuted = weight1
        else:
            weight1_permuted = weight1
        
        # Compute cost matrix (negative dot product for maximization)
        # We're matching output neurons, so cost matrix is [out2 x out1]
        cost_matrix = -torch.mm(weight2, weight1_permuted.t())
        
        # Solve assignment problem
        row_ind, col_ind = linear_sum_assignment(cost_matrix.numpy())
        
        # Create permutation matrix for output dimension
        n = weight1.shape[0]
        perm = torch.zeros(n, n)
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
        
        # Track the output permutation from each block
        prev_output_perm = None
        
        for block_idx, (source_block, result_block) in enumerate(
            zip(weights.blocks, result.blocks)):
            
            block_perm = perm_spec.block_perms[block_idx]
            d_model = source_block.attention.qkv_weight.shape[1]
            
            # Handle attention permutations
            if block_perm['attention_out'] is not None:
                P_attn = block_perm['attention_out']
                
                # Apply attention permutation to QKV weights
                # QKV: [3*d_model, d_model] 
                # We need to permute the output dimension (rows) with block diagonal P
                P_attn_expanded = torch.block_diag(P_attn, P_attn, P_attn)
                
                if prev_output_perm is not None:
                    # Account for previous block's output permutation on input
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
                
                # Permute projection to unpermute output
                # proj: [d_model, d_model]
                result_block.attention.proj_weight = torch.mm(
                    result_block.attention.proj_weight, P_attn.t()
                )
                
                # Attention output is now unpermuted (identity)
            
            # Handle MLP permutations
            if len(result_block.mlp_weights) >= 2 and block_perm['mlp1'] is not None:
                P_mlp1 = block_perm['mlp1']
                
                # MLP structure in ViT:
                # fc1: [hidden_dim, d_model] - expands from d_model to hidden_dim
                # fc2: [d_model, hidden_dim] - projects back to d_model
                
                # First MLP layer
                if prev_output_perm is not None:
                    # Apply P_mlp1 to output dim, prev_perm^T to input dim
                    result_block.mlp_weights = list(result_block.mlp_weights)
                    result_block.mlp_weights[0] = torch.mm(
                        torch.mm(P_mlp1, result_block.mlp_weights[0]),
                        prev_output_perm.t()
                    )
                else:
                    result_block.mlp_weights = list(result_block.mlp_weights)
                    result_block.mlp_weights[0] = torch.mm(P_mlp1, result_block.mlp_weights[0])
                
                # Permute first layer bias
                if len(result_block.mlp_biases) > 0:
                    result_block.mlp_biases = list(result_block.mlp_biases)
                    result_block.mlp_biases[0] = torch.mv(P_mlp1, result_block.mlp_biases[0])
                
                # Second MLP layer: apply P_mlp1^T to input dimension
                # This unpermutes the hidden dimension
                result_block.mlp_weights[1] = torch.mm(
                    result_block.mlp_weights[1], P_mlp1.t()
                )
                
                result_block.mlp_weights = tuple(result_block.mlp_weights)
                result_block.mlp_biases = tuple(result_block.mlp_biases)
                
                # Output is back to d_model dimension, unpermuted
                prev_output_perm = torch.eye(d_model)
            else:
                # No MLP permutation, propagate identity
                prev_output_perm = torch.eye(d_model)
        
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
                # Reference stays the same
                canonicalized.append(reference)
                print(f"Model {i} (reference): No changes")
            else:
                print(f"Canonicalizing model {i} to match reference...")
                
                # Save original weights for comparison (only for model 1)
                if i == 1:
                    orig_block0_attn = model.blocks[0].attention.qkv_weight.clone()
                    orig_block0_mlp0 = model.blocks[0].mlp_weights[0].clone() if len(model.blocks[0].mlp_weights) > 0 else None
                
                # Initialize permutation spec
                perm_spec = PermutationSpec(len(model.blocks))
                current_model = copy.deepcopy(model)
                
                # Iterative refinement (Algorithm 1 from paper)
                for iteration in range(self.num_iterations):
                    print(f"  Iteration {iteration + 1}/{self.num_iterations}")
                    
                    # Track permutation dimension through the network
                    current_dim_perm = None
                    
                    # Process each block
                    for block_idx in range(len(current_model.blocks)):
                        current_block = current_model.blocks[block_idx]
                        reference_block = reference.blocks[block_idx]
                        
                        # Get embedding dimension from attention
                        d_model = current_block.attention.qkv_weight.shape[1]
                        
                        # Match attention heads
                        inter_perm, intra_perms, composed_perm = self.match_attention_heads(
                            current_block.attention, reference_block.attention
                        )
                        perm_spec.set_block_perm(block_idx, 'attention_out', composed_perm)
                        
                        # Match MLP layers
                        if len(current_block.mlp_weights) >= 1:
                            mlp1_perm = self.match_mlp_layer(
                                current_block.mlp_weights[0],
                                reference_block.mlp_weights[0],
                                prev_perm=current_dim_perm
                            )
                            perm_spec.set_block_perm(block_idx, 'mlp1', mlp1_perm)
                            
                            current_dim_perm = torch.eye(d_model)
                    
                    # Apply permutations
                    current_model = self.apply_permutation_to_weights(model, perm_spec)
                
                # Print weight comparison for model 1, block 0
                if i == 1:
                    print("\n  === Model 1, Block 0 Weight Comparison ===")
                    print("  Attention QKV weights:")
                    print(f"    Original shape: {orig_block0_attn.shape}")
                    print(f"    Original first 5x5 values:\n{orig_block0_attn[:5, :5]}")
                    print(f"    Canonical shape: {current_model.blocks[0].attention.qkv_weight.shape}")
                    print(f"    Canonical first 5x5 values:\n{current_model.blocks[0].attention.qkv_weight[:5, :5]}")
                    
                    # Check if weights actually changed
                    attn_diff = torch.norm(current_model.blocks[0].attention.qkv_weight - orig_block0_attn).item()
                    print(f"    Weight difference norm: {attn_diff:.6f}")
                    
                    if orig_block0_mlp0 is not None:
                        print("\n  MLP fc1 weights:")
                        print(f"    Original shape: {orig_block0_mlp0.shape}")
                        print(f"    Original first 5x5 values:\n{orig_block0_mlp0[:5, :5]}")
                        print(f"    Canonical shape: {current_model.blocks[0].mlp_weights[0].shape}")
                        print(f"    Canonical first 5x5 values:\n{current_model.blocks[0].mlp_weights[0][:5, :5]}")
                        
                        mlp_diff = torch.norm(current_model.blocks[0].mlp_weights[0] - orig_block0_mlp0).item()
                        print(f"    Weight difference norm: {mlp_diff:.6f}")
                    
                    print("  " + "="*50 + "\n")
                
                # Check overall weight change
                orig_flat = model.flatten()
                canon_flat = current_model.flatten()
                weight_change = torch.norm(canon_flat - orig_flat).item()
                print(f"  Final weight change magnitude: {weight_change:.4f}")
                
                canonicalized.append(current_model)
        
        return canonicalized


class GitReBasinMatcher:
    """
    Alternative matcher using Git Re-Basin approach for comparison
    """
    
    def __init__(self, num_iterations: int = 10):
        self.num_iterations = num_iterations
    
    def match_weights(self,
                     model1: VisionTransformerWeightSpace,
                     model2: VisionTransformerWeightSpace) -> PermutationSpec:
        """
        Iterative weight matching following Git Re-Basin
        """
        perm_spec = PermutationSpec(len(model1.blocks))
        
        # Simplified implementation
        # Would iterate through layers and compute optimal permutations
        
        return perm_spec