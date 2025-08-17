import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from typing import List
import random
import argparse
from tqdm import tqdm

from .vit_models import create_vit_tiny, VisionTransformer
from .weight_space import VisionTransformerWeightSpace
from .permutation_matching import TransFusionMatcher
from .visualization import WeightSpaceVisualizer
from .evaluation import compare_models_performance, evaluate_interpolation_path
from .data_utils import get_cifar10_dataloaders


def train_vit(model: VisionTransformer,
             train_loader,
             val_loader,
             epochs: int = 10,
             lr: float = 1e-3,
             device: torch.device = torch.device('cpu')) -> VisionTransformer:
    """
    Train a Vision Transformer model
    """
    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()
    
    best_val_acc = 0
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = output.max(1)
            train_total += target.size(0)
            train_correct += predicted.eq(target).sum().item()
            
            if batch_idx % 10 == 0:
                pbar.set_postfix({
                    'Loss': f'{train_loss/(batch_idx+1):.3f}',
                    'Acc': f'{100.*train_correct/train_total:.1f}%'
                })
        
        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                _, predicted = output.max(1)
                val_total += target.size(0)
                val_correct += predicted.eq(target).sum().item()
        
        val_acc = 100. * val_correct / val_total
        print(f'Epoch {epoch+1}: Val Acc = {val_acc:.2f}%')
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
        
        scheduler.step()
    
    print(f'Best validation accuracy: {best_val_acc:.2f}%')
    return model


def main():
    parser = argparse.ArgumentParser(description='ViT Weight Space Canonicalization')
    parser.add_argument('--num_models', type=int, default=3, 
                       help='Number of ViT models to train')
    parser.add_argument('--epochs', type=int, default=10,
                       help='Number of training epochs per model')
    parser.add_argument('--batch_size', type=int, default=128,
                       help='Batch size for training')
    parser.add_argument('--lr', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to use for training')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--train_from_scratch', action='store_true',
                       help='Train models from scratch instead of using pre-trained')
    
    args = parser.parse_args()
    
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Get data loaders
    print("Loading CIFAR-10 dataset...")
    train_loader, val_loader, test_loader = get_cifar10_dataloaders(
        batch_size=args.batch_size
    )
    
    # Create or load models
    models = []
    
    if args.train_from_scratch:
        print(f"\nTraining {args.num_models} ViT models from scratch...")
        for i in range(args.num_models):
            print(f"\n{'='*50}")
            print(f"Training Model {i+1}/{args.num_models}")
            print('='*50)
            
            # Create model with different initialization
            torch.manual_seed(args.seed + i)
            model = create_vit_tiny(num_classes=10)
            
            # Train model
            model = train_vit(model, train_loader, val_loader, 
                            epochs=args.epochs, lr=args.lr, device=device)
            models.append(model)
            
            # Save model
            torch.save(model.state_dict(), f'src/vit_weight_canonicalization/vit_models/vit_model_{i}.pth')
            print(f"Saved model {i} to src/vit_weight_canonicalization/vit_models/vit_model_{i}.pth")
    else:
        print(f"\nLoading {args.num_models} pre-trained ViT models...")
        for i in range(args.num_models):
            model = create_vit_tiny(num_classes=10)
            try:
                model.load_state_dict(torch.load(f'src/vit_weight_canonicalization/vit_models/vit_model_{i}.pth'))
                print(f"Loaded model {i}")
            except:
                print(f"Could not load model {i}, initializing randomly")
                torch.manual_seed(args.seed + i)
                model = create_vit_tiny(num_classes=10)
            models.append(model)
    
    # Extract weight spaces
    print("\nExtracting weight spaces from models...")
    weight_spaces = []
    for i, model in enumerate(models):
        ws = VisionTransformerWeightSpace.from_vit_model(model)
        weight_spaces.append(ws)
        print(f"Extracted weight space from model {i}")
    
    # Perform canonicalization
    print("\nPerforming weight-matching canonicalization...")
    matcher = TransFusionMatcher()
    canonical_weight_spaces = matcher.canonicalize_model(
        weight_spaces, reference_idx=0
    )
    
    # Create canonicalized models
    canonical_models = []
    for i, ws in enumerate(canonical_weight_spaces):
        model = create_vit_tiny(num_classes=10)
        ws.apply_to_model(model)
        canonical_models.append(model)
    
    # Evaluate and compare performance
    print("\nEvaluating model performance...")
    results = compare_models_performance(
        models, canonical_models, test_loader, device
    )
    
    # Visualize weight spaces
    print("\nVisualizing weight spaces...")
    visualizer = WeightSpaceVisualizer()
    
    # PCA visualization
    fig_pca = visualizer.plot_pca_comparison(
        weight_spaces, canonical_weight_spaces,
        labels=[f"Model {i}" for i in range(len(models))]
    )
    plt.savefig('pca_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # t-SNE visualization
    print("Computing t-SNE (this may take a while)...")
    fig_tsne = visualizer.plot_tsne_comparison(
        weight_spaces, canonical_weight_spaces,
        labels=[f"Model {i}" for i in range(len(models))]
    )
    plt.savefig('tsne_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Weight distance heatmap
    fig_dist_orig = visualizer.plot_weight_distances(
        weight_spaces,
        labels=[f"Model {i}" for i in range(len(models))]
    )
    plt.savefig('weight_distances_original.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    fig_dist_canon = visualizer.plot_weight_distances(
        canonical_weight_spaces,
        labels=[f"Model {i}" for i in range(len(models))]
    )
    plt.savefig('weight_distances_canonical.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Evaluate interpolation paths
    if len(models) >= 2:
        print("\nEvaluating interpolation paths...")
        
        # Original models
        print("Interpolating between original models 0 and 1...")
        orig_interp = evaluate_interpolation_path(
            models[0], models[1], test_loader, num_points=11, device=device
        )
        
        # Canonical models
        print("Interpolating between canonical models 0 and 1...")
        canon_interp = evaluate_interpolation_path(
            canonical_models[0], canonical_models[1], test_loader, 
            num_points=11, device=device
        )
        
        # Plot interpolation results
        fig, ax = plt.subplots(figsize=(10, 6))
        
        alphas = [r['alpha'] for r in orig_interp]
        orig_accs = [r['accuracy'] for r in orig_interp]
        canon_accs = [r['accuracy'] for r in canon_interp]
        
        ax.plot(alphas, orig_accs, 'o-', label='Original', linewidth=2)
        ax.plot(alphas, canon_accs, 's-', label='Canonicalized', linewidth=2)
        ax.set_xlabel('Interpolation coefficient (α)', fontsize=12)
        ax.set_ylabel('Test Accuracy (%)', fontsize=12)
        ax.set_title('Linear Mode Connectivity', fontsize=14)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        plt.savefig('interpolation_comparison.png', dpi=150, bbox_inches='tight')
        plt.show()
    
    print("\nExperiment complete!")
    print("Generated visualizations:")
    print("  - pca_comparison.png")
    print("  - tsne_comparison.png") 
    print("  - weight_distances_original.png")
    print("  - weight_distances_canonical.png")
    if len(models) >= 2:
        print("  - interpolation_comparison.png")


if __name__ == "__main__":
    main()

# python -m src.vit_weight_canonicalization.main --num_models 10 --epochs 5 --seed 42 --train_from_scratch
# python -m src.vit_weight_canonicalization.main --num_models 10 --seed 42 

# git push origin HEAD:ViT-Rebasin-and-Flow-Matching