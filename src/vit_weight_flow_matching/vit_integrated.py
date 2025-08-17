"""
Complete Integrated Pipeline
1. Train/load ViT models
2. Canonicalize them
3. Use canonicalized models for flow matching
4. Generate new models
5. Evaluate and compare
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
from typing import List, Dict
import json
from datetime import datetime

# Import all our modules
from src.vit_weight_canonicalization.vit_models import create_vit_tiny, VisionTransformer
from src.vit_weight_canonicalization.weight_space import VisionTransformerWeightSpace
from src.vit_weight_canonicalization.permutation_matching import TransFusionMatcher
from src.vit_weight_canonicalization.data_utils import get_cifar10_dataloaders
from src.vit_weight_canonicalization.evaluation import evaluate_model
from src.vit_weight_flow_matching.vit_flow_matching import ViTFlowMatcher, SimpleCFM, TimeConditionedMLP

class IntegratedPipeline:
    """Complete pipeline from loading pre-trained models to generation"""
    
    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device)
        
        # Create directories
        self.setup_directories()
        
        # Storage for all models
        self.original_models = []
        self.canonical_weight_spaces = []
        self.generated_weight_spaces = []
        self.all_results = {}
        
    def setup_directories(self):
        """Create necessary directories"""
        dirs = ['checkpoints', 'results', 'visualizations']
        for d in dirs:
            os.makedirs(d, exist_ok=True)
        
        # Note: We don't create src/vit_weight_canonicalization/vit_models since we're loading from it
    
    def step1_prepare_vit_models(self):
        """Step 1: Load pre-trained ViT models from src/vit_weight_canonicalization/vit_models folder"""
        print("\n" + "="*60)
        print("STEP 1: Loading Pre-trained ViT Models")
        print("="*60)
        
        # Get test loader for evaluation
        _, _, test_loader = get_cifar10_dataloaders(
            batch_size=self.args.batch_size
        )
        
        # Check if src/vit_weight_canonicalization/vit_models directory exists
        if not os.path.exists('src/vit_weight_canonicalization/vit_models'):
            raise FileNotFoundError(
                "src/vit_weight_canonicalization/vit_models directory not found! Please ensure pre-trained models are in src/vit_weight_canonicalization/vit_models/"
            )
        
        # Load available models
        models_loaded = 0
        for i in range(self.args.num_models):
            model_path = f'src/vit_weight_canonicalization/vit_models/vit_model_{i}.pth'
            
            if os.path.exists(model_path):
                print(f"Loading model {i} from {model_path}...")
                model = create_vit_tiny()
                
                # Load with map_location for compatibility
                state_dict = torch.load(model_path, map_location=self.device)
                model.load_state_dict(state_dict)
                
                self.original_models.append(model)
                
                # Evaluate original model
                metrics = evaluate_model(model, test_loader, self.device)
                print(f"  Model {i} - Acc: {metrics['accuracy']:.2f}%, Loss: {metrics['loss']:.4f}")
                models_loaded += 1
            else:
                print(f"Warning: Model file {model_path} not found, skipping...")
        
        if models_loaded == 0:
            raise FileNotFoundError(
                "No pre-trained models found in src/vit_weight_canonicalization/vit_models/! "
                "Expected files like vit_model_0.pth, vit_model_1.pth, etc."
            )
        
        print(f"\nSuccessfully loaded {models_loaded} pre-trained models")
        
        # Update num_models to reflect actual loaded models
        if models_loaded < self.args.num_models:
            print(f"Note: Only {models_loaded} models found, adjusting num_models accordingly")
            self.args.num_models = models_loaded
        
        return test_loader
    
    def step2_canonicalize_models(self):
        """Step 2: Canonicalize models using TransFusion"""
        print("\n" + "="*60)
        print("STEP 2: Canonicalizing Models")
        print("="*60)
        
        # Extract weight spaces
        original_weight_spaces = []
        for model in self.original_models:
            ws = VisionTransformerWeightSpace.from_vit_model(model)
            original_weight_spaces.append(ws)
        
        # Perform canonicalization
        matcher = TransFusionMatcher(num_iterations=self.args.canon_iters)
        self.canonical_weight_spaces = matcher.canonicalize_model(
            original_weight_spaces, 
            reference_idx=0
        )
        
        # Save canonical models
        for i, ws in enumerate(self.canonical_weight_spaces):
            model = create_vit_tiny(num_classes=10)
            ws.apply_to_model(model)
            torch.save(model.state_dict(), f'src/vit_weight_canonicalization/vit_models/canonical_model_{i}.pth')
        
        # Verify canonicalization
        print("\nVerifying canonicalization...")
        for i in range(1, len(self.canonical_weight_spaces)):
            orig_flat = original_weight_spaces[i].flatten()
            canon_flat = self.canonical_weight_spaces[i].flatten()
            
            weight_diff = torch.norm(canon_flat - orig_flat).item()
            print(f"  Model {i} weight change: {weight_diff:.4f}")
    
    def step3_train_flow_matching(self, test_loader):
        """Step 3: Train flow matching model"""
        print("\n" + "="*60)
        print("STEP 3: Training Flow Matching Model")
        print("="*60)
        
        # Initialize flow matching pipeline
        self.flow_pipeline = ViTFlowMatcher(
            canonical_weight_spaces=self.canonical_weight_spaces,
            original_weight_spaces=self.canonical_weight_spaces  # Using canonical for both
        )
        
        # Train flow model
        training_metrics = self.flow_pipeline.train_flow(
            n_iters=self.args.flow_iters,
            lr=self.args.flow_lr
        )
        
        # Save training metrics
        with open('results/flow_training_metrics.json', 'w') as f:
            json.dump({k: v for k, v in training_metrics.items() if v}, f, indent=2)
        
        return training_metrics
    
    def step4_generate_models(self):
        """Step 4: Generate new models using flow matching"""
        print("\n" + "="*60)
        print("STEP 4: Generating New Models")
        print("="*60)
        
        self.generated_weight_spaces = self.flow_pipeline.generate_vits(
            n_samples=self.args.num_generate,
            n_steps=self.args.generation_steps
        )
        
        # Save generated models
        for i, ws in enumerate(self.generated_weight_spaces):
            model = create_vit_tiny(num_classes=10)
            ws.apply_to_model(model)
            torch.save(model.state_dict(), f'src/vit_weight_canonicalization/vit_models/generated_model_{i}.pth')
        
        print(f"Generated and saved {len(self.generated_weight_spaces)} models")
    
    def step5_evaluate_all(self, test_loader):
        """Step 5: Comprehensive evaluation"""
        print("\n" + "="*60)
        print("STEP 5: Comprehensive Evaluation")
        print("="*60)
        
        # Evaluate all model types
        results = {
            'original': [],
            'canonical': [],
            'generated': []
        }
        
        # Original models
        print("\nEvaluating original models...")
        for i, model in enumerate(self.original_models):
            metrics = evaluate_model(model, test_loader, self.device)
            results['original'].append(metrics)
            print(f"  Original {i}: Acc={metrics['accuracy']:.2f}%, Loss={metrics['loss']:.4f}")
        
        # Canonical models
        print("\nEvaluating canonical models...")
        for i, ws in enumerate(self.canonical_weight_spaces):
            model = create_vit_tiny(num_classes=10)
            ws.apply_to_model(model)
            metrics = evaluate_model(model, test_loader, self.device)
            results['canonical'].append(metrics)
            print(f"  Canonical {i}: Acc={metrics['accuracy']:.2f}%, Loss={metrics['loss']:.4f}")
        
        # Generated models
        print("\nEvaluating generated models...")
        for i, ws in enumerate(self.generated_weight_spaces):
            model = create_vit_tiny(num_classes=10)
            ws.apply_to_model(model)
            metrics = evaluate_model(model, test_loader, self.device)
            results['generated'].append(metrics)
            print(f"  Generated {i}: Acc={metrics['accuracy']:.2f}%, Loss={metrics['loss']:.4f}")
        
        # Compute and print statistics
        self.all_results = results
        self.print_statistics(results)
        
        # Save results
        with open('results/evaluation_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        return results
    
    def print_statistics(self, results):
        """Print comprehensive statistics"""
        print("\n" + "="*60)
        print("SUMMARY STATISTICS")
        print("="*60)
        
        for model_type in ['original', 'canonical', 'generated']:
            if model_type in results and results[model_type]:
                accs = [r['accuracy'] for r in results[model_type]]
                losses = [r['loss'] for r in results[model_type]]
                
                print(f"\n{model_type.upper()} Models:")
                print(f"  Count: {len(accs)}")
                print(f"  Accuracy: {np.mean(accs):.2f}% ± {np.std(accs):.2f}%")
                print(f"  Range: [{np.min(accs):.2f}%, {np.max(accs):.2f}%]")
                print(f"  Loss: {np.mean(losses):.4f} ± {np.std(losses):.4f}")
    
    def step6_visualize_results(self):
        """Step 6: Create comprehensive visualizations"""
        print("\n" + "="*60)
        print("STEP 6: Creating Visualizations")
        print("="*60)
        
        # Weight space visualization
        self.flow_pipeline.visualize_weight_distributions()
        
        # Diversity analysis
        self.flow_pipeline.analyze_diversity()
        
        # Performance comparison plot
        self.plot_performance_comparison()
        
        print("Visualizations saved to visualizations/ directory")
    
    def plot_performance_comparison(self):
        """Create performance comparison plots"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Accuracy comparison
        model_types = ['original', 'canonical', 'generated']
        colors = ['blue', 'green', 'red']
        
        for i, (model_type, color) in enumerate(zip(model_types, colors)):
            if model_type in self.all_results:
                accs = [r['accuracy'] for r in self.all_results[model_type]]
                positions = np.random.normal(i, 0.04, size=len(accs))
                ax1.scatter(positions, accs, alpha=0.6, s=100, c=color, label=model_type.capitalize())
        
        ax1.set_xticks([0, 1, 2])
        ax1.set_xticklabels(['Original', 'Canonical', 'Generated'])
        ax1.set_ylabel('Test Accuracy (%)')
        ax1.set_title('Model Performance Comparison')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Loss comparison
        for i, (model_type, color) in enumerate(zip(model_types, colors)):
            if model_type in self.all_results:
                losses = [r['loss'] for r in self.all_results[model_type]]
                positions = np.random.normal(i, 0.04, size=len(losses))
                ax2.scatter(positions, losses, alpha=0.6, s=100, c=color)
        
        ax2.set_xticks([0, 1, 2])
        ax2.set_xticklabels(['Original', 'Canonical', 'Generated'])
        ax2.set_ylabel('Test Loss')
        ax2.set_title('Loss Comparison')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('visualizations/performance_comparison.png', dpi=150)
        plt.show()
    
    def run_complete_pipeline(self):
        """Run the complete integrated pipeline"""
        print("\n" + "="*60)
        print("COMPLETE INTEGRATED PIPELINE")
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*60)
        
        # Step 1: Prepare models
        test_loader = self.step1_prepare_vit_models()
        
        # Step 2: Canonicalize
        self.step2_canonicalize_models()
        
        # Step 3: Train flow matching
        self.step3_train_flow_matching(test_loader)
        
        # Step 4: Generate new models
        self.step4_generate_models()
        
        # Step 5: Evaluate all
        results = self.step5_evaluate_all(test_loader)
        
        # Step 6: Visualize
        self.step6_visualize_results()
        
        print("\n" + "="*60)
        print("PIPELINE COMPLETE!")
        print("="*60)
        
        return results


def main():
    parser = argparse.ArgumentParser(description='Integrated ViT Canonicalization + Flow Matching Pipeline')
    
    # Model loading args
    parser.add_argument('--num_models', type=int, default=5,
                       help='Number of ViT models to load (will load vit_model_0.pth through vit_model_N.pth)')
    parser.add_argument('--batch_size', type=int, default=128,
                       help='Batch size for evaluation')
    
    # Canonicalization args
    parser.add_argument('--canon_iters', type=int, default=3,
                       help='Iterations for canonicalization')
    
    # Flow matching args
    parser.add_argument('--flow_iters', type=int, default=500,
                       help='Flow matching training iterations')
    parser.add_argument('--flow_lr', type=float, default=1e-3)
    parser.add_argument('--num_generate', type=int, default=5,
                       help='Number of models to generate')
    parser.add_argument('--generation_steps', type=int, default=50,
                       help='Integration steps for generation')
    
    # General args
    parser.add_argument('--device', type=str, 
                       default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    
    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Check if src/vit_weight_canonicalization/vit_models directory exists
    if not os.path.exists('src/vit_weight_canonicalization/vit_models'):
        print("ERROR: src/vit_weight_canonicalization/vit_models directory not found!")
        print("Please ensure you have pre-trained ViT models in the src/vit_weight_canonicalization/vit_models/ folder")
        print("Expected files: vit_model_0.pth, vit_model_1.pth, etc.")
        return
    
    # List available models
    available_models = [f for f in os.listdir('src/vit_weight_canonicalization/vit_models') if f.startswith('vit_model_') and f.endswith('.pth')]
    print(f"\nFound {len(available_models)} pre-trained models in src/vit_weight_canonicalization/vit_models/:")
    for model_file in sorted(available_models)[:10]:  # Show first 10
        print(f"  - {model_file}")
    if len(available_models) > 10:
        print(f"  ... and {len(available_models) - 10} more")
    
    # Run pipeline
    pipeline = IntegratedPipeline(args)
    results = pipeline.run_complete_pipeline()
    
    print("\nAll results saved to results/ directory")
    print("Visualizations saved to visualizations/ directory")
    print("Models saved to src/vit_weight_canonicalization/vit_models/ directory")


if __name__ == "__main__":
    main()

# python -m src.vit_weight_flow_matching.vit_integrated --num_models 10 --flow_iters 200 --canon_iters 5 --num_generate 5 --generation_steps 100
# python -m src.vit_weight_flow_matching.vit_integrated --num_models 5 --flow_iters 200 --canon_iters 5 --num_generate 5 --generation_steps 100
