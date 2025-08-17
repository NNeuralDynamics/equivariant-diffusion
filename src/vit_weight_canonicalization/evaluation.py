import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import List, Dict, Tuple
import numpy as np
from tqdm import tqdm
from .weight_space import VisionTransformerWeightSpace
from .vit_models import VisionTransformer


def evaluate_model(model: nn.Module, 
                  dataloader: DataLoader, 
                  device: torch.device = torch.device('cpu')) -> Dict[str, float]:
    """
    Evaluate a model on a dataset
    
    Returns:
        Dictionary with accuracy and loss
    """
    model.eval()
    model.to(device)
    
    total_correct = 0
    total_samples = 0
    total_loss = 0.0
    
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(tqdm(dataloader, desc="Evaluating")):
            data, target = data.to(device), target.to(device)
            
            # Forward pass
            output = model(data)
            loss = F.cross_entropy(output, target)
            
            # Calculate accuracy
            pred = output.argmax(dim=1, keepdim=True)
            correct = pred.eq(target.view_as(pred)).sum().item()
            
            total_correct += correct
            total_samples += data.size(0)
            total_loss += loss.item()
    
    accuracy = 100.0 * total_correct / total_samples
    avg_loss = total_loss / len(dataloader)
    
    return {
        'accuracy': accuracy,
        'loss': avg_loss
    }


def compare_models_performance(original_models: List[nn.Module],
                              canonical_models: List[nn.Module],
                              test_loader: DataLoader,
                              device: torch.device = torch.device('cpu')) -> Dict:
    """
    Compare performance of original and canonicalized models
    """
    results = {
        'original': [],
        'canonical': []
    }
    
    print("Evaluating original models...")
    for i, model in enumerate(original_models):
        print(f"  Model {i}...")
        metrics = evaluate_model(model, test_loader, device)
        results['original'].append(metrics)
    
    print("\nEvaluating canonicalized models...")
    for i, model in enumerate(canonical_models):
        print(f"  Model {i}...")
        metrics = evaluate_model(model, test_loader, device)
        results['canonical'].append(metrics)
    
    # Print summary
    print("\n" + "="*50)
    print("Performance Summary")
    print("="*50)
    
    print("\nOriginal Models:")
    for i, metrics in enumerate(results['original']):
        print(f"  Model {i}: Acc={metrics['accuracy']:.2f}%, Loss={metrics['loss']:.4f}")
    
    print("\nCanonicalized Models:")
    for i, metrics in enumerate(results['canonical']):
        print(f"  Model {i}: Acc={metrics['accuracy']:.2f}%, Loss={metrics['loss']:.4f}")
    
    # Compute statistics
    orig_accs = [m['accuracy'] for m in results['original']]
    canon_accs = [m['accuracy'] for m in results['canonical']]
    
    print(f"\nOriginal - Mean Acc: {np.mean(orig_accs):.2f}% ± {np.std(orig_accs):.2f}%")
    print(f"Canonical - Mean Acc: {np.mean(canon_accs):.2f}% ± {np.std(canon_accs):.2f}%")
    
    return results


def interpolate_weights(weight1: VisionTransformerWeightSpace,
                       weight2: VisionTransformerWeightSpace,
                       alpha: float) -> VisionTransformerWeightSpace:
    """
    Linear interpolation between two weight spaces
    """
    import copy
    result = copy.deepcopy(weight1)
    
    # Interpolate patch embedding
    result.patch_embed_weight = (1 - alpha) * weight1.patch_embed_weight + alpha * weight2.patch_embed_weight
    if result.patch_embed_bias is not None:
        result.patch_embed_bias = (1 - alpha) * weight1.patch_embed_bias + alpha * weight2.patch_embed_bias
    
    # Interpolate tokens and embeddings
    result.cls_token = (1 - alpha) * weight1.cls_token + alpha * weight2.cls_token
    result.pos_embed = (1 - alpha) * weight1.pos_embed + alpha * weight2.pos_embed
    
    # Interpolate transformer blocks
    for i, (block1, block2, result_block) in enumerate(zip(weight1.blocks, weight2.blocks, result.blocks)):
        # Attention weights
        result_block.attention.qkv_weight = (1 - alpha) * block1.attention.qkv_weight + alpha * block2.attention.qkv_weight
        if result_block.attention.qkv_bias is not None:
            result_block.attention.qkv_bias = (1 - alpha) * block1.attention.qkv_bias + alpha * block2.attention.qkv_bias
        result_block.attention.proj_weight = (1 - alpha) * block1.attention.proj_weight + alpha * block2.attention.proj_weight
        if result_block.attention.proj_bias is not None:
            result_block.attention.proj_bias = (1 - alpha) * block1.attention.proj_bias + alpha * block2.attention.proj_bias
        
        # Layer norms
        result_block.norm1_weight = (1 - alpha) * block1.norm1_weight + alpha * block2.norm1_weight
        result_block.norm1_bias = (1 - alpha) * block1.norm1_bias + alpha * block2.norm1_bias
        result_block.norm2_weight = (1 - alpha) * block1.norm2_weight + alpha * block2.norm2_weight
        result_block.norm2_bias = (1 - alpha) * block1.norm2_bias + alpha * block2.norm2_bias
        
        # MLP weights
        result_block.mlp_weights = tuple(
            (1 - alpha) * w1 + alpha * w2 
            for w1, w2 in zip(block1.mlp_weights, block2.mlp_weights)
        )
        result_block.mlp_biases = tuple(
            (1 - alpha) * b1 + alpha * b2 
            for b1, b2 in zip(block1.mlp_biases, block2.mlp_biases)
        )
    
    # Interpolate final norm and head
    result.norm_weight = (1 - alpha) * weight1.norm_weight + alpha * weight2.norm_weight
    result.norm_bias = (1 - alpha) * weight1.norm_bias + alpha * weight2.norm_bias
    result.head_weight = (1 - alpha) * weight1.head_weight + alpha * weight2.head_weight
    result.head_bias = (1 - alpha) * weight1.head_bias + alpha * weight2.head_bias
    
    return result


def evaluate_interpolation_path(model1: nn.Module,
                               model2: nn.Module,
                               test_loader: DataLoader,
                               num_points: int = 11,
                               device: torch.device = torch.device('cpu')) -> Dict:
    """
    Evaluate models along interpolation path
    """
    alphas = np.linspace(0, 1, num_points)
    results = []
    
    # Extract weight spaces
    ws1 = VisionTransformerWeightSpace.from_vit_model(model1)
    ws2 = VisionTransformerWeightSpace.from_vit_model(model2)
    
    for alpha in tqdm(alphas, desc="Evaluating interpolation"):
        # Interpolate weights
        ws_interp = interpolate_weights(ws1, ws2, alpha)
        
        # Create new model with SAME architecture as original models
        # Use the exact same configuration
        model_interp = type(model1)(
            img_size=model1.patch_embed.img_size,
            patch_size=model1.patch_embed.patch_size,
            embed_dim=model1.embed_dim,
            depth=len(model1.blocks),
            num_heads=model1.blocks[0].attn.num_heads,
            num_classes=model1.num_classes
        )
        
        ws_interp.apply_to_model(model_interp)
        
        # Evaluate
        metrics = evaluate_model(model_interp, test_loader, device)
        results.append({
            'alpha': alpha,
            'accuracy': metrics['accuracy'],
            'loss': metrics['loss']
        })
    
    return results