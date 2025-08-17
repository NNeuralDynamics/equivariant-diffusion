"""
ViT Flow Matching Pipeline
Combines weight space canonicalization with flow matching to generate new ViT models
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Optional
import os
import logging
from tqdm import tqdm
import traceback
from collections import namedtuple

# Import from our modules
from src.vit_weight_canonicalization.weight_space import VisionTransformerWeightSpace
from src.vit_weight_canonicalization.vit_models import create_vit_tiny, VisionTransformer
from src.vit_weight_canonicalization.permutation_matching import TransFusionMatcher
from src.vit_weight_canonicalization.evaluation import evaluate_model
from src.vit_weight_canonicalization.data_utils import get_cifar10_dataloaders

# Flow matching components (from your provided code)
Bunch = namedtuple('Bunch', ['t', 'x0', 'xt', 'x1', 'ut', 'eps', 'lambda_t', 'batch_size'])

class WeightSpaceDataset(Dataset):
    """Dataset for flattened weight spaces"""
    
    def __init__(self, weight_spaces: List[VisionTransformerWeightSpace], 
                 augment: bool = False, noise_scale: float = 0.01):
        """
        Args:
            weight_spaces: List of VisionTransformerWeightSpace objects
            augment: Whether to add noise augmentation
            noise_scale: Scale of noise if augmenting
        """
        self.weight_spaces = weight_spaces
        self.augment = augment
        self.noise_scale = noise_scale
        
        # Flatten all weight spaces
        self.flattened_weights = []
        for ws in weight_spaces:
            flat = ws.flatten()
            self.flattened_weights.append(flat)
        
        # Stack into tensor
        self.data = torch.stack(self.flattened_weights)
        self.dim = self.data.shape[1]
        
        print(f"Created dataset with {len(self.data)} samples, dimension {self.dim}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx].clone()
        
        if self.augment:
            # Add small noise for augmentation
            noise = torch.randn_like(sample) * self.noise_scale
            sample = sample + noise
        
        return sample, idx  # Return index as label for tracking


class ViTFlowMatcher:
    """Main pipeline for ViT generation via flow matching"""
    
    def __init__(self, 
                 canonical_weight_spaces: List[VisionTransformerWeightSpace],
                 original_weight_spaces: List[VisionTransformerWeightSpace],
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Args:
            canonical_weight_spaces: Canonicalized ViT weight spaces
            original_weight_spaces: Original ViT weight spaces for comparison
        """
        self.canonical_weight_spaces = canonical_weight_spaces
        self.original_weight_spaces = original_weight_spaces
        self.device = torch.device(device)
        
        # Get dimension from first weight space
        self.weight_dim = canonical_weight_spaces[0].flatten().shape[0]
        print(f"Weight dimension: {self.weight_dim}")
        
        # Create datasets
        self.create_datasets()
        
        # Initialize flow model
        self.init_flow_model()
        
        # Storage for generated models
        self.generated_weight_spaces = []
        self.generation_metrics = {}
    
    def create_datasets(self, batch_size: int = 4):
        """Create source and target dataloaders for flow matching"""
        
        # Create canonical dataset (target distribution)
        self.canonical_dataset = WeightSpaceDataset(
            self.canonical_weight_spaces, 
            augment=True, 
            noise_scale=0.01
        )
        
        # For flow matching, we need source (noise) and target (canonical models)
        # Source: Gaussian noise with same dimension
        noise_data = torch.randn(len(self.canonical_weight_spaces), self.weight_dim)
        noise_dataset = TensorDataset(noise_data)
        
        # Create dataloaders
        self.source_loader = DataLoader(
            noise_dataset, 
            batch_size=batch_size, 
            shuffle=True,
            drop_last=True
        )
        
        self.target_loader = DataLoader(
            self.canonical_dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True
        )
        
        print(f"Created dataloaders with batch size {batch_size}")
    
    def init_flow_model(self, hidden_dims: List[int] = None):
        """Initialize the flow matching model"""
        if hidden_dims is None:
            # Scale hidden dimensions based on weight dimension
            if self.weight_dim < 10000:
                hidden_dims = [512, 512, 512]
            elif self.weight_dim < 50000:
                hidden_dims = [1024, 1024, 512]
            else:
                hidden_dims = [2048, 1024, 512]
        
        self.flow_model = TimeConditionedMLP(
            input_dim=self.weight_dim,
            hidden_dims=hidden_dims,
            output_dim=self.weight_dim
        ).to(self.device)
        
        # Initialize flow matcher
        self.cfm = SimpleCFM(
            sourceloader=self.source_loader,
            targetloader=self.target_loader,
            model=self.flow_model,
            fm_type="vanilla",
            mode="velocity",
            device=self.device
        )
        
        print(f"Initialized flow model with hidden dims: {hidden_dims}")
        print(f"Total parameters: {sum(p.numel() for p in self.flow_model.parameters())}")
    
    def train_flow(self, n_iters: int = 1000, lr: float = 1e-3):
        """Train the flow matching model"""
        print("\nTraining flow matching model...")
        
        optimizer = optim.AdamW(self.flow_model.parameters(), lr=lr)
        self.cfm.train(n_iters=n_iters, optimizer=optimizer, log_freq=10)
        
        # Plot training metrics
        self.cfm.plot_metrics()
        
        return self.cfm.metrics
    
    def generate_vits(self, n_samples: int = 5, n_steps: int = 50) -> List[VisionTransformerWeightSpace]:
        """Generate new ViT weight spaces using the trained flow model"""
        print(f"\nGenerating {n_samples} new ViT models...")
        
        # Start from Gaussian noise
        x0 = torch.randn(n_samples, self.weight_dim).to(self.device)
        
        # Map through flow to generate weights
        generated_flat = self.cfm.map(x0, n_steps=n_steps, return_traj=False)
        
        # Convert back to weight spaces
        generated_weight_spaces = []
        reference_ws = self.canonical_weight_spaces[0]  # Use first as template
        
        for i in range(n_samples):
            # Create new weight space with generated weights
            ws = self._unflatten_to_weight_space(
                generated_flat[i], 
                reference_ws
            )
            generated_weight_spaces.append(ws)
            self.generated_weight_spaces.append(ws)
        
        print(f"Generated {len(generated_weight_spaces)} new weight spaces")
        return generated_weight_spaces
    
    def _unflatten_to_weight_space(self, 
                                   flat_weights: torch.Tensor,
                                   template: VisionTransformerWeightSpace) -> VisionTransformerWeightSpace:
        """Convert flattened weights back to VisionTransformerWeightSpace"""
        import copy
        
        # Create a copy of the template
        new_ws = copy.deepcopy(template)
        
        # Unflatten weights following the same order as flatten()
        idx = 0
        
        # Patch embedding
        patch_embed_size = new_ws.patch_embed_weight.numel()
        new_ws.patch_embed_weight = flat_weights[idx:idx+patch_embed_size].reshape_as(new_ws.patch_embed_weight)
        idx += patch_embed_size
        
        if new_ws.patch_embed_bias is not None:
            bias_size = new_ws.patch_embed_bias.numel()
            new_ws.patch_embed_bias = flat_weights[idx:idx+bias_size].reshape_as(new_ws.patch_embed_bias)
            idx += bias_size
        
        # Class token
        cls_size = new_ws.cls_token.numel()
        new_ws.cls_token = flat_weights[idx:idx+cls_size].reshape_as(new_ws.cls_token)
        idx += cls_size
        
        # Position embedding
        pos_size = new_ws.pos_embed.numel()
        new_ws.pos_embed = flat_weights[idx:idx+pos_size].reshape_as(new_ws.pos_embed)
        idx += pos_size
        
        # Transformer blocks
        for block in new_ws.blocks:
            # Attention QKV
            qkv_size = block.attention.qkv_weight.numel()
            block.attention.qkv_weight = flat_weights[idx:idx+qkv_size].reshape_as(block.attention.qkv_weight)
            idx += qkv_size
            
            if block.attention.qkv_bias is not None:
                bias_size = block.attention.qkv_bias.numel()
                block.attention.qkv_bias = flat_weights[idx:idx+bias_size].reshape_as(block.attention.qkv_bias)
                idx += bias_size
            
            # Attention projection
            proj_size = block.attention.proj_weight.numel()
            block.attention.proj_weight = flat_weights[idx:idx+proj_size].reshape_as(block.attention.proj_weight)
            idx += proj_size
            
            if block.attention.proj_bias is not None:
                bias_size = block.attention.proj_bias.numel()
                block.attention.proj_bias = flat_weights[idx:idx+bias_size].reshape_as(block.attention.proj_bias)
                idx += bias_size
            
            # Layer norms
            for norm_weight, norm_bias in [(block.norm1_weight, block.norm1_bias),
                                           (block.norm2_weight, block.norm2_bias)]:
                weight_size = norm_weight.numel()
                norm_weight.data = flat_weights[idx:idx+weight_size].reshape_as(norm_weight)
                idx += weight_size
                
                bias_size = norm_bias.numel()
                norm_bias.data = flat_weights[idx:idx+bias_size].reshape_as(norm_bias)
                idx += bias_size
            
            # MLP weights and biases
            for w in block.mlp_weights:
                w_size = w.numel()
                w.data = flat_weights[idx:idx+w_size].reshape_as(w)
                idx += w_size
            
            for b in block.mlp_biases:
                b_size = b.numel()
                b.data = flat_weights[idx:idx+b_size].reshape_as(b)
                idx += b_size
        
        # Final norm and head
        norm_weight_size = new_ws.norm_weight.numel()
        new_ws.norm_weight = flat_weights[idx:idx+norm_weight_size].reshape_as(new_ws.norm_weight)
        idx += norm_weight_size
        
        norm_bias_size = new_ws.norm_bias.numel()
        new_ws.norm_bias = flat_weights[idx:idx+norm_bias_size].reshape_as(new_ws.norm_bias)
        idx += norm_bias_size
        
        head_weight_size = new_ws.head_weight.numel()
        new_ws.head_weight = flat_weights[idx:idx+head_weight_size].reshape_as(new_ws.head_weight)
        idx += head_weight_size
        
        head_bias_size = new_ws.head_bias.numel()
        new_ws.head_bias = flat_weights[idx:idx+head_bias_size].reshape_as(new_ws.head_bias)
        idx += head_bias_size
        
        return new_ws
    
    def evaluate_generated(self, test_loader, device='cuda'):
        """Evaluate generated models and compare with training models"""
        print("\nEvaluating generated models...")
        
        results = {
            'canonical': [],
            'generated': []
        }
        
        # Evaluate canonical models
        print("Evaluating canonical models...")
        for i, ws in enumerate(self.canonical_weight_spaces):
            model = create_vit_tiny(num_classes=10)
            ws.apply_to_model(model)
            metrics = evaluate_model(model, test_loader, device)
            results['canonical'].append(metrics)
            print(f"  Canonical {i}: Acc={metrics['accuracy']:.2f}%, Loss={metrics['loss']:.4f}")
        
        # Evaluate generated models
        print("\nEvaluating generated models...")
        for i, ws in enumerate(self.generated_weight_spaces):
            model = create_vit_tiny(num_classes=10)
            ws.apply_to_model(model)
            metrics = evaluate_model(model, test_loader, device)
            results['generated'].append(metrics)
            print(f"  Generated {i}: Acc={metrics['accuracy']:.2f}%, Loss={metrics['loss']:.4f}")
        
        # Compute statistics
        self.generation_metrics = self._compute_statistics(results)
        
        return results
    
    def _compute_statistics(self, results):
        """Compute statistics comparing canonical and generated models"""
        stats = {}
        
        for key in ['canonical', 'generated']:
            accs = [r['accuracy'] for r in results[key]]
            losses = [r['loss'] for r in results[key]]
            
            stats[key] = {
                'mean_acc': np.mean(accs),
                'std_acc': np.std(accs),
                'min_acc': np.min(accs),
                'max_acc': np.max(accs),
                'mean_loss': np.mean(losses),
                'std_loss': np.std(losses)
            }
        
        print("\n" + "="*50)
        print("Performance Statistics")
        print("="*50)
        
        for key in ['canonical', 'generated']:
            print(f"\n{key.capitalize()} Models:")
            s = stats[key]
            print(f"  Accuracy: {s['mean_acc']:.2f}% ± {s['std_acc']:.2f}%")
            print(f"  Range: [{s['min_acc']:.2f}%, {s['max_acc']:.2f}%]")
            print(f"  Loss: {s['mean_loss']:.4f} ± {s['std_loss']:.4f}")
        
        return stats
    
    def visualize_weight_distributions(self):
        """Visualize weight space distributions"""
        from sklearn.decomposition import PCA
        
        # Flatten all weight spaces
        canonical_flat = [ws.flatten().numpy() for ws in self.canonical_weight_spaces]
        generated_flat = [ws.flatten().numpy() for ws in self.generated_weight_spaces]
        
        # Combine for PCA
        all_weights = np.vstack(canonical_flat + generated_flat)
        
        # Apply PCA
        pca = PCA(n_components=2)
        all_transformed = pca.fit_transform(all_weights)
        
        # Split back
        n_canonical = len(canonical_flat)
        canonical_pca = all_transformed[:n_canonical]
        generated_pca = all_transformed[n_canonical:]
        
        # Plot
        fig, ax = plt.subplots(figsize=(10, 8))
        
        ax.scatter(canonical_pca[:, 0], canonical_pca[:, 1], 
                  label='Canonical', alpha=0.7, s=100, c='blue')
        ax.scatter(generated_pca[:, 0], generated_pca[:, 1], 
                  label='Generated', alpha=0.7, s=100, c='red')
        
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')
        ax.set_title('Weight Space Distribution: Canonical vs Generated')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('weight_distribution_comparison.png', dpi=150)
        plt.show()
        
        return fig
    
    def analyze_diversity(self):
        """Analyze diversity of generated models"""
        
        def compute_pairwise_distances(weight_spaces):
            n = len(weight_spaces)
            distances = np.zeros((n, n))
            
            for i in range(n):
                for j in range(i+1, n):
                    w1 = weight_spaces[i].flatten()
                    w2 = weight_spaces[j].flatten()
                    dist = torch.norm(w1 - w2).item()
                    distances[i, j] = dist
                    distances[j, i] = dist
            
            return distances
        
        # Compute distances
        canonical_dists = compute_pairwise_distances(self.canonical_weight_spaces)
        generated_dists = compute_pairwise_distances(self.generated_weight_spaces)
        
        # Plot distributions
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Canonical distances
        sns.heatmap(canonical_dists, ax=ax1, cmap='viridis', annot=True, fmt='.1f')
        ax1.set_title('Pairwise Distances: Canonical Models')
        
        # Generated distances
        sns.heatmap(generated_dists, ax=ax2, cmap='viridis', annot=True, fmt='.1f')
        ax2.set_title('Pairwise Distances: Generated Models')
        
        plt.tight_layout()
        plt.savefig('diversity_analysis.png', dpi=150)
        plt.show()
        
        # Compute diversity metrics
        canonical_diversity = np.mean(canonical_dists[np.triu_indices_from(canonical_dists, k=1)])
        generated_diversity = np.mean(generated_dists[np.triu_indices_from(generated_dists, k=1)])
        
        print(f"\nDiversity Analysis:")
        print(f"  Canonical mean distance: {canonical_diversity:.2f}")
        print(f"  Generated mean distance: {generated_diversity:.2f}")
        print(f"  Diversity ratio (gen/canon): {generated_diversity/canonical_diversity:.2f}")
        
        return canonical_dists, generated_dists


# Include the flow matching components from your provided code
class ResidualBlock(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)
        self.activation = nn.ReLU()
        self.residual = (in_dim == out_dim)

    def forward(self, x):
        out = self.norm(self.activation(self.linear(x)))
        if self.residual:
            return out + x
        return out


class TimeConditionedMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super().__init__()
        self.input_dim = input_dim + 1  # +1 for time
        
        layers = []
        prev_dim = self.input_dim
        for hidden_dim in hidden_dims:
            layers.append(ResidualBlock(prev_dim, hidden_dim))
            prev_dim = hidden_dim

        self.hidden_layers = nn.Sequential(*layers)
        self.output_layer = nn.Linear(prev_dim, output_dim)

    def forward(self, inputs, t):
        x = torch.cat([inputs, t], dim=1)
        x = self.hidden_layers(x)
        return self.output_layer(x)


# Include SimpleCFM class from your code
class SimpleCFM:
    def __init__(
        self,
        sourceloader,
        targetloader,
        model,
        fm_type="vanilla",
        mode="velocity",
        t_dist="uniform",
        device=None,
        normalize_pred=False,
        geometric=False,
    ):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
            
        self.t_dist = t_dist
        self.sourceloader = sourceloader
        self.targetloader = targetloader
        self.model = model
        self.fm_type = fm_type
        self.mode = mode
        self.sigma = 0.001
        self.normalize_pred = normalize_pred
        self.geometric = geometric
        self.metrics = {"train_loss": [], "time": [], "grad_norm": [], "flow_norm": [], "true_norm": []}
        self.best_loss = float('inf')
        self.best_model_state = None
    
    def sample_from_loader(self, loader):
        """Sample from a dataloader with proper error handling"""
        try:
            if not hasattr(loader, '_iterator') or loader._iterator is None:
                loader._iterator = iter(loader)
            try:
                batch = next(loader._iterator)
            except StopIteration:
                loader._iterator = iter(loader)
                batch = next(loader._iterator)
            return batch[0]  # Return just the tensor, not the tuple
        except Exception as e:
            logging.info(f"Error sampling from loader: {str(e)}")
            # Return a default tensor if sampling fails
            return torch.zeros(loader.batch_size, loader.dataset[0][0].shape[0], device=self.device)

    def sample_time_and_flow(self):
        """Sample time, start and end points, and intermediate x_t"""
        x0 = self.sample_from_loader(self.sourceloader)
        x1 = self.sample_from_loader(self.targetloader)
        
        # Ensure consistent batch size
        batch_size = min(x0.size(0), x1.size(0))
        x0 = x0[:batch_size].to(self.device)
        x1 = x1[:batch_size].to(self.device)
        
        if self.t_dist == "uniform":
            t = torch.rand(batch_size).to(self.device)
        elif self.t_dist == "beta":
            alpha, beta = torch.tensor(1.0), torch.tensor(2.0)
            t = torch.distributions.Beta(alpha, beta).sample((batch_size,)).to(self.device)
        
        t_pad = t.reshape(-1, *([1] * (x0.dim() - 1)))
        
        mu_t = (1 - t_pad) * x0 + t_pad * x1
        sigma_pad = torch.tensor(self.sigma).reshape(-1, *([1] * (x0.dim() - 1))).to(self.device)
        xt = mu_t + sigma_pad * torch.randn_like(x0).to(self.device)
        ut = x1 - x0
        
        t = t.unsqueeze(-1)
        
        return Bunch(t=t, x0=x0, xt=xt, x1=x1, ut=ut, eps=0, lambda_t=0, batch_size=batch_size)
    
    def forward(self, flow):
        """Forward pass through the model with proper error handling"""
        try:
            # Forward pass directly through the model
            flow_pred = self.model(flow.xt, flow.t)
            return None, flow_pred
        except Exception as e:
            logging.info(f"Error in forward pass: {str(e)}")
            traceback.print_exc()
            # Return zero tensors as fallback
            return None, torch.zeros_like(flow.ut)
    
    def loss_fn(self, flow_pred, flow):
        """Compute loss between predicted and true flows"""
        if self.mode == "target":
            l_flow = torch.mean((flow_pred.squeeze() - flow.x1) ** 2)
        elif self.mode == "velocity":
            l_flow = torch.mean((flow_pred.squeeze() - flow.ut) ** 2)
        else:
            # Fallback to velocity mode if unknown
            l_flow = torch.mean((flow_pred.squeeze() - flow.ut) ** 2)
        return None, l_flow
    
    def map(self, x0, n_steps=20, return_traj=False, noise_scale=0.001):
        """Map points using the flow model to generate new weights"""
        if self.best_model_state is not None:
            current_state = {k: v.clone() for k, v in self.model.state_dict().items()}
            self.model.load_state_dict(self.best_model_state)

        self.model.eval()

        batch_size, flat_dim = x0.size()
        traj = [] if return_traj else None

        # Create time steps for Euler integration
        times = torch.linspace(0, 1, n_steps).to(self.device)
        dt = times[1] - times[0]

        # Initialize result with starting point
        xt = x0.clone()

        for t in times[:-1]: 
            if return_traj:
                traj.append(xt.detach().clone())

            with torch.no_grad():
                # Create time tensor with correct shape
                t_tensor = torch.ones(batch_size, 1).to(self.device) * t

                try:
                    pred = self.model(xt, t_tensor)

                    if pred.dim() > 2:
                        pred = pred.squeeze(-1)

                    if self.mode == "velocity":
                        vt = pred
                    else:  # mode == "target"
                        vt = pred - xt

                    xt = xt + vt * dt

                    if t > 0.8:
                        xt = xt + torch.randn_like(xt) * noise_scale
                        
                except Exception as e:
                    logging.info(f"Error during mapping at t={t}: {str(e)}")

        if return_traj:
            traj.append(xt.detach().clone())

        if self.best_model_state is not None:
            self.model.load_state_dict(current_state)
            
        self.model.train()

        return traj if return_traj else xt
    
    def train(self, n_iters=10, optimizer=None, sigma=0.001, patience=1e99, log_freq=5):
        """Train the flow model"""
        self.sigma = sigma
        self.metrics = {"train_loss": [], "time": [], "grad_norm": [], "flow_norm": [], "true_norm": []}
        last_loss = 1e99
        patience_count = 0
        
        pbar = tqdm(range(n_iters), desc="Training steps")
        for i in pbar:
            try:
                optimizer.zero_grad()
                
                flow = self.sample_time_and_flow()
                _, flow_pred = self.forward(flow)
                _, loss = self.loss_fn(flow_pred, flow)
                
                if not torch.isnan(loss) and not torch.isinf(loss):
                    loss.backward()
                    optimizer.step()

                    if i % 100 == 0 or i == n_iters - 1:
                        print(f"\n[Iter {i}], Loss = {loss.item():.6f}")
                        
                        checkpoint_dir = 'checkpoints'
                        os.makedirs(checkpoint_dir, exist_ok=True)
                        ckpt_path = os.path.join(checkpoint_dir, f'flow_model_iter_{i}.pth')
                        torch.save(self.model.state_dict(), ckpt_path)
                
                    # Save best model
                    if loss.item() < self.best_loss:
                        self.best_loss = loss.item()
                        self.best_model_state = {k: v.clone() for k, v in self.model.state_dict().items()}
                else:
                    logging.info(f"Skipping step {i} due to invalid loss: {loss.item()}")
                    continue
                
                # Early stopping
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
                    
                    pbar.set_description(f"Loss {train_loss_val:.6f}, ∇ norm {grad_norm:.6f}")
            
            except Exception as e:
                logging.info(f"Error during training iteration {i}: {str(e)}")
                traceback.print_exc()
                break
    
    def get_grad_norm(self):
        """Compute gradient norm"""
        total = 0
        for p in self.model.parameters():
            if p.grad is not None:
                param_norm = p.grad.detach().data.norm(2)
                total += param_norm.item() ** 2
        total = total**0.5
        return total
    
    def plot_metrics(self):
        """Plot training metrics"""
        labels = list(self.metrics.keys())
        lists = list(self.metrics.values())
        n = len(lists)
        
        fig, axs = plt.subplots(1, n, figsize=(3 * n, 3))
        for i, (label, lst) in enumerate(zip(labels, lists)):
            if len(lst) > 0:  # Only plot if we have data
                axs[i].plot(lst)
                axs[i].grid()
                axs[i].title.set_text(label)
                if label == "train_loss":
                    axs[i].set_yscale("log")
        plt.tight_layout()
        plt.savefig('flow_training_metrics.png', dpi=150)
        plt.show()


def main_flow_matching_pipeline():
    """Main function to run the complete pipeline"""
    
    # Load pre-trained models from vit_models folder
    print("Loading pre-trained ViT models from vit_models/...")
    
    canonical_weight_spaces = []
    original_weight_spaces = []
    
    # Check if vit_models directory exists
    if not os.path.exists('vit_models'):
        raise FileNotFoundError(
            "vit_models directory not found! "
            "Please ensure pre-trained models are in vit_models/"
        )
    
    # Find all available model files
    model_files = sorted([f for f in os.listdir('vit_models') 
                         if f.startswith('vit_model_') and f.endswith('.pth')])
    
    if not model_files:
        raise FileNotFoundError(
            "No pre-trained models found in vit_models/! "
            "Expected files like vit_model_0.pth, vit_model_1.pth, etc."
        )
    
    print(f"Found {len(model_files)} pre-trained models")
    
    # Load models
    for model_file in model_files[:10]:  # Limit to first 10 models
        model_path = os.path.join('src/vit_weight_canonicalization/vit_models', model_file)
        print(f"Loading {model_file}...")
        
        model = create_vit_tiny(num_classes=10)
        state_dict = torch.load(model_path, map_location='cpu')
        model.load_state_dict(state_dict)
        
        ws = VisionTransformerWeightSpace.from_vit_model(model)
        canonical_weight_spaces.append(ws)
        original_weight_spaces.append(ws)
    
    print(f"Loaded {len(canonical_weight_spaces)} models successfully")
    
    # Perform canonicalization
    print("\nCanonicalizing models...")
    matcher = TransFusionMatcher(num_iterations=3)
    canonical_weight_spaces = matcher.canonicalize_model(
        canonical_weight_spaces, 
        reference_idx=0
    )
    
    # Initialize flow matching pipeline
    flow_pipeline = ViTFlowMatcher(
        canonical_weight_spaces=canonical_weight_spaces,
        original_weight_spaces=original_weight_spaces
    )
    
    # Train flow matching model
    print("\nTraining flow matching model...")
    flow_pipeline.train_flow(n_iters=1000, lr=1e-3)
    
    # Generate new ViT models
    print("\nGenerating new models...")
    generated_weight_spaces = flow_pipeline.generate_vits(n_samples=5, n_steps=50)
    
    # Evaluate generated models
    print("\nEvaluating models...")
    _, _, test_loader = get_cifar10_dataloaders(batch_size=128)
    results = flow_pipeline.evaluate_generated(test_loader)
    
    # Visualize results
    print("\nCreating visualizations...")
    flow_pipeline.visualize_weight_distributions()
    flow_pipeline.analyze_diversity()
    
    # Save generated models
    print("\nSaving generated models...")
    for i, ws in enumerate(generated_weight_spaces):
        model = create_vit_tiny(num_classes=10)
        ws.apply_to_model(model)
        torch.save(model.state_dict(), f'vit_models/generated_model_{i}.pth')
        print(f"  Saved generated_model_{i}.pth")
    
    return flow_pipeline, results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    flow_pipeline, results = main_flow_matching_pipeline()