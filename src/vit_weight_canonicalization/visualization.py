import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from typing import List, Dict, Optional, Tuple
from .weight_space import VisionTransformerWeightSpace


class WeightSpaceVisualizer:
    """Visualization tools for weight spaces"""
    
    def __init__(self, figsize: Tuple[int, int] = (15, 5)):
        self.figsize = figsize
        sns.set_style("whitegrid")
        
    def plot_pca_comparison(self, 
                           original_weights: List[VisionTransformerWeightSpace],
                           canonical_weights: List[VisionTransformerWeightSpace],
                           labels: Optional[List[str]] = None,
                           n_components: int = 2):
        """
        Compare original and canonicalized weights using PCA
        """
        if labels is None:
            labels = [f"Model {i}" for i in range(len(original_weights))]
        
        # Flatten weights
        orig_flat = [w.flatten().numpy() for w in original_weights]
        canon_flat = [w.flatten().numpy() for w in canonical_weights]
        
        # Combine for PCA
        all_weights_orig = np.vstack(orig_flat)
        all_weights_canon = np.vstack(canon_flat)
        
        # Fit PCA
        pca_orig = PCA(n_components=n_components)
        pca_canon = PCA(n_components=n_components)
        
        orig_transformed = pca_orig.fit_transform(all_weights_orig)
        canon_transformed = pca_canon.fit_transform(all_weights_canon)
        
        # Plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.figsize)
        
        # Original weights
        for i, label in enumerate(labels):
            ax1.scatter(orig_transformed[i, 0], orig_transformed[i, 1], 
                       label=label, s=100)
        ax1.set_title("Original Weight Space (PCA)")
        ax1.set_xlabel(f"PC1 ({pca_orig.explained_variance_ratio_[0]:.2%})")
        ax1.set_ylabel(f"PC2 ({pca_orig.explained_variance_ratio_[1]:.2%})")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Canonicalized weights
        for i, label in enumerate(labels):
            ax2.scatter(canon_transformed[i, 0], canon_transformed[i, 1], 
                       label=label, s=100)
        ax2.set_title("Canonicalized Weight Space (PCA)")
        ax2.set_xlabel(f"PC1 ({pca_canon.explained_variance_ratio_[0]:.2%})")
        ax2.set_ylabel(f"PC2 ({pca_canon.explained_variance_ratio_[1]:.2%})")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_tsne_comparison(self,
                           original_weights: List[VisionTransformerWeightSpace],
                           canonical_weights: List[VisionTransformerWeightSpace],
                           labels: Optional[List[str]] = None,
                           perplexity: int = 30,
                           n_iter: int = 500):
        """
        Compare original and canonicalized weights using t-SNE
        """
        if labels is None:
            labels = [f"Model {i}" for i in range(len(original_weights))]
        
        # Sample weights if too large (t-SNE is computationally expensive)
        orig_flat = []
        canon_flat = []
        
        for w_orig, w_canon in zip(original_weights, canonical_weights):
            orig_vec = w_orig.flatten().numpy()
            canon_vec = w_canon.flatten().numpy()
            
            # Sample if too large
            if len(orig_vec) > 10000:
                indices = np.random.choice(len(orig_vec), 10000, replace=False)
                orig_vec = orig_vec[indices]
                canon_vec = canon_vec[indices]
            
            orig_flat.append(orig_vec)
            canon_flat.append(canon_vec)
        
        # Stack weights
        all_weights_orig = np.vstack(orig_flat)
        all_weights_canon = np.vstack(canon_flat)
        
        # Fit t-SNE
        tsne = TSNE(n_components=2, perplexity=min(perplexity, len(all_weights_orig)-1), 
                   n_iter_without_progress=n_iter, random_state=42)
        
        orig_embedded = tsne.fit_transform(all_weights_orig)
        canon_embedded = tsne.fit_transform(all_weights_canon)
        
        # Plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.figsize)
        
        # Original weights
        for i, label in enumerate(labels):
            ax1.scatter(orig_embedded[i, 0], orig_embedded[i, 1], 
                       label=label, s=100)
        ax1.set_title("Original Weight Space (t-SNE)")
        ax1.set_xlabel("t-SNE 1")
        ax1.set_ylabel("t-SNE 2")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Canonicalized weights
        for i, label in enumerate(labels):
            ax2.scatter(canon_embedded[i, 0], canon_embedded[i, 1], 
                       label=label, s=100)
        ax2.set_title("Canonicalized Weight Space (t-SNE)")
        ax2.set_xlabel("t-SNE 1")
        ax2.set_ylabel("t-SNE 2")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_weight_distances(self,
                            weights: List[VisionTransformerWeightSpace],
                            labels: Optional[List[str]] = None):
        """
        Plot pairwise distances between weight spaces
        """
        if labels is None:
            labels = [f"Model {i}" for i in range(len(weights))]
        
        n = len(weights)
        distance_matrix = np.zeros((n, n))
        
        # Compute pairwise distances
        for i in range(n):
            for j in range(n):
                if i != j:
                    w1 = weights[i].flatten()
                    w2 = weights[j].flatten()
                    distance_matrix[i, j] = torch.norm(w1 - w2).item()
        
        # Plot heatmap
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(distance_matrix, annot=True, fmt='.2f', 
                   xticklabels=labels, yticklabels=labels,
                   cmap='coolwarm', ax=ax)
        ax.set_title("Pairwise Weight Space Distances")
        plt.tight_layout()
        
        return fig