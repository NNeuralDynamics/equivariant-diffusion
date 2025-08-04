import sys

sys.path.insert(1, "..")

import torch
import numpy as np
from collections import defaultdict
from typing import NamedTuple
from scipy.optimize import linear_sum_assignment
from nn.relational_transformer import RelationalTransformer
from nn.graph_constructor import GraphConstructor
from flow.flow_matching import CFM
from tqdm import tqdm
import copy
import logging
from sklearn.decomposition import KernelPCA

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.basicConfig(stream=sys.stdout, format='%(asctime)s %(levelname)s: %(message)s', level=logging.INFO, datefmt='%I:%M:%S')

class PermutationSpec:
    def __init__(self, perm_to_axes: dict, axes_to_perm: dict):
        self.perm_to_axes = perm_to_axes
        self.axes_to_perm = axes_to_perm


def permutation_spec_from_axes_to_perm(axes_to_perm: dict) -> PermutationSpec:
    perm_to_axes = defaultdict(list)
    for wk, axis_perms in axes_to_perm.items():
        for axis, perm in enumerate(axis_perms):
            if perm is not None:
                perm_to_axes[perm].append((wk, axis))
    return PermutationSpec(dict(perm_to_axes), axes_to_perm)


def convnet_permutation_spec() -> PermutationSpec:
    conv = lambda name, p_in, p_out: {
        f"{name}.weight": (p_out, p_in, None, None),
        f"{name}.bias": (p_out,)
    }
    return permutation_spec_from_axes_to_perm({
        **conv("conv1", None, "P1"),
        **conv("conv2", "P1", "P2"),
        **conv("conv3", "P2", "P3"),
        **conv("head.0", "P3", "P4"),
        **conv("head.2", "P4", None),
    })


def get_permuted_param(ps: PermutationSpec, perm, k: str, params, except_axis=None):
    w = params[k]
    for axis, p in enumerate(ps.axes_to_perm[k]):
        if axis == except_axis:
            continue
        if p is not None:
            w = np.take(w, perm[p], axis=axis)
    return w


def apply_permutation(ps: PermutationSpec, perm, params):
    return {k: get_permuted_param(ps, perm, k, params) for k in params.keys()}


def weight_matching(rng: np.random.Generator,
                    ps: PermutationSpec,
                    params_a,
                    params_b,
                    max_iter=100,
                    init_perm=None,
                    silent=True):
    perm_sizes = {p: params_a[axes[0][0]].shape[axes[0][1]] for p, axes in ps.perm_to_axes.items()}
    perm = {p: np.arange(n) for p, n in perm_sizes.items()} if init_perm is None else init_perm
    perm_names = list(perm.keys())

    for iteration in range(max_iter):
        progress = False
        rng.shuffle(perm_names)
        for p in perm_names:
            n = perm_sizes[p]
            A = np.zeros((n, n))
            for wk, axis in ps.perm_to_axes[p]:
                w_a = params_a[wk]
                w_b = get_permuted_param(ps, perm, wk, params_b, except_axis=axis)
                w_a = np.moveaxis(w_a, axis, 0).reshape((n, -1))
                w_b = np.moveaxis(w_b, axis, 0).reshape((n, -1))
                A += w_a @ w_b.T

            ri, ci = linear_sum_assignment(A, maximize=True)
            oldL = np.sum(A[np.arange(n), perm[p]])
            newL = np.sum(A[np.arange(n), ci])
            if not silent:
                print(f"{iteration}/{p}: {newL - oldL}")
            progress = progress or newL > oldL + 1e-12
            perm[p] = ci
        if not progress:
            break

    return perm


def get_permuted_models_data(
    ref_point=0,
    model_dir="imagenet_convnet_models",
    num_models=239,
    device=device
):
    """Apply JAX-based weight matching to align models with a reference model."""
    
    ref_model = ConvNet()
    ref_model_path = f"{model_dir}/convnet_weights_{ref_point}.pt"
    
    try:
        print(device)
        ref_model.load_state_dict(torch.load(ref_model_path, map_location=device))
        ref_model = ref_model.to(device)
        logging.info(f"Loaded reference model from {ref_model_path}")
    except Exception as e:
        logging.error(f"Failed to load reference model: {e}")
        raise e
    
    ps = convnet_permutation_spec()
    
    # Extract params and buffers from ref model (PyTorch)
    params_a = {k: v.clone().detach() for k, v in ref_model.state_dict().items() 
               if k in ps.axes_to_perm}
    
    # Convert ref params to JAX arrays
    params_a_np = {k: v.cpu().numpy() for k, v in params_a.items()}

    permuted_models = []
    org_models = []

    for i in tqdm(range(num_models), desc="Processing models"):
        if i == ref_point:
            continue
        
        model_path = f"{model_dir}/convnet_weights_{i}.pt"
        if not os.path.exists(model_path):
            logging.info(f"Skipping model {i} - file not found")
            continue
        
        try:
            # Load model B
            model_b = ConvNet()
            model_b.load_state_dict(torch.load(model_path, map_location=device))
            model_b = model_b.to(device)
            org_models.append(model_b)
            params_b = {k: v.clone().detach() for k, v in model_b.state_dict().items() 
                       if k in ps.axes_to_perm}
            params_b_np = {k: v.cpu().numpy() for k, v in params_b.items()}
            rng = np.random.default_rng(123 + i)
            perm = weight_matching(rng, ps, params_a_np, params_b_np)
            permuted_params_b_np = apply_permutation(ps, perm, params_b_np)
            permuted_params_b_torch = {k: torch.tensor(v) for k, v in permuted_params_b_np.items()}

            
            # Create a new model copy and update state dict
            reconstructed_model = copy.deepcopy(model_b)
            state_dict = reconstructed_model.state_dict()
            
            for k in permuted_params_b_torch:
                state_dict[k] = permuted_params_b_torch[k]
            
            reconstructed_model.load_state_dict(state_dict)
            reconstructed_model = reconstructed_model.to(device)
            permuted_models.append(reconstructed_model)
        
        except Exception as e:
            logging.error(f"Error processing model {i}: {e}")
            continue
        
        torch.cuda.empty_cache()
    
    print(f"Processed {len(permuted_models)} models successfully")
    return ref_model, org_models, permuted_models


import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from torch.optim import Adam, SGD
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torchvision import transforms, datasets
import torchvision
import copy
from collections import defaultdict
import os
import traceback

class ConvNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, stride=2, padding=1)   # 224 → 112
        self.conv2 = nn.Conv2d(32, 64, 3, stride=2, padding=1)  # 112 → 56
        self.conv3 = nn.Conv2d(64, 128, 3, stride=2, padding=1) # 56 → 28

        # Replaces linear layers with conv layers (fully convolutional)
        self.head = nn.Sequential(
            nn.Conv2d(128, 512, kernel_size=3, stride=2, padding=1),  # 28 → 14
            nn.ReLU(),
            nn.Conv2d(512, num_classes, kernel_size=1)  # Output: (num_classes × 14 × 14)
        )

    def forward(self, x):
        x = F.relu(self.conv1(x))  # (B, 32, 112, 112)
        x = F.relu(self.conv2(x))  # (B, 64, 56, 56)
        x = F.relu(self.conv3(x))  # (B, 128, 28, 28)
        x = self.head(x)           # (B, num_classes, 14, 14)
        return x.mean(dim=[2, 3])

# Flatten and rebuild
class WeightSpaceObject:
    def __init__(self, weights, biases):
        self.weights = tuple(weights)
        self.biases = tuple(biases)
    def flatten(self, device=None):
        flat = torch.cat([w.flatten() for w in self.weights] + [b.flatten() for b in self.biases])
        return flat.to(device) if device else flat
    @classmethod
    def from_flat(cls, flat, weight_shapes, bias_shapes, device=None):
        sizes = [np.prod(s) for s in weight_shapes + bias_shapes]
        parts = []
        start = 0
        for size in sizes:
            parts.append(flat[start:start+size])
            start += size
        weights = [parts[i].reshape(weight_shapes[i]) for i in range(len(weight_shapes))]
        biases = [parts[len(weight_shapes) + i].reshape(bias_shapes[i]) for i in range(len(bias_shapes))]
        return cls(weights, biases).to(device)
    def to(self, device):
        return WeightSpaceObject([w.to(device) for w in self.weights],
                                 [b.to(device) for b in self.biases])

# Simple Bunch class for storing data
class Bunch:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

# Safe deflatten function that checks bounds before accessing tensors
def safe_deflatten(flat, batch_size, starts, ends):
    """Safely deflatten a tensor without index errors"""
    parts = []
    actual_batch_size = flat.size(0)
    
    # Ensure we don't exceed the actual batch size
    safe_batch_size = min(actual_batch_size, batch_size)
    
    for i in range(safe_batch_size):
        batch_parts = []
        for si, ei in zip(starts, ends):
            if si < ei:  # Only process valid ranges
                batch_parts.append(flat[i][si:ei])
        parts.append(batch_parts)
    
    return parts

# Conditional Flow Matching class
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
        
        # Generate intermediate points and velocity field
        mu_t = (1 - t_pad) * x0 + t_pad * x1
        sigma_pad = torch.tensor(self.sigma).reshape(-1, *([1] * (x0.dim() - 1))).to(self.device)
        xt = mu_t + sigma_pad * torch.randn_like(x0).to(self.device)
        ut = x1 - x0
        
        # Reshape t to match model expectations
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
        elif self.fm_type == "ot":
            l_flow = torch.mean((flow_pred.squeeze() - flow.ut) ** 2)
        else:
            # Fallback to velocity mode if unknown
            l_flow = torch.mean((flow_pred.squeeze() - flow.ut) ** 2)
        return None, l_flow
    
    def map(self, x0, n_steps=20, return_traj=False, noise_scale=0.001):
        """Map points using the flow model to generate new weights"""
        # Use the best model state if available
        if self.best_model_state is not None:
            # Store current state
            current_state = {k: v.clone() for k, v in self.model.state_dict().items()}
            # Load best state
            self.model.load_state_dict(self.best_model_state)

        # Set the model to evaluation mode
        self.model.eval()

        batch_size, flat_dim = x0.size()
        traj = [] if return_traj else None

        # Create time steps for Euler integration
        times = torch.linspace(0, 1, n_steps).to(self.device)
        dt = times[1] - times[0]  # Time step size

        # Initialize result with starting point
        xt = x0.clone()

        for t in times[:-1]:  # Don't push forward at t=1
            if return_traj:
                traj.append(xt.detach().clone())

            with torch.no_grad():
                # Create time tensor with correct shape
                t_tensor = torch.ones(batch_size, 1).to(self.device) * t

                # Get model prediction
                try:
                    pred = self.model(xt, t_tensor)

                    # Make sure prediction has the right shape
                    if pred.dim() > 2:
                        pred = pred.squeeze(-1)

                    # Calculate velocity based on mode
                    if self.mode == "velocity":
                        vt = pred
                    else:  # mode == "target"
                        vt = pred - xt

                    # Euler integration step
                    xt = xt + vt * dt

                    # Add small noise at later timesteps to prevent mode collapse
                    if t > 0.8:
                        xt = xt + torch.randn_like(xt) * noise_scale
                        
                except Exception as e:
                    logging.info(f"Error during mapping at t={t}: {str(e)}")
                    # Skip this step if there's an error

        # Add final point to trajectory if tracking
        if return_traj:
            traj.append(xt.detach().clone())

        # Restore original model state if we used the best state
        if self.best_model_state is not None:
            self.model.load_state_dict(current_state)
            
        # Return to training mode
        self.model.train()

        return traj if return_traj else xt
       
    
    def vector_field(self, xt, t):
        """Compute vector field at point xt and time t"""
        # Forward pass through model
        _, pred = self.forward(Bunch(xt=xt, t=t, batch_size=xt.size(0)))
        
        if self.mode == "velocity":
            vt = pred
        elif self.mode == "target":
            vt = pred - xt
        
        return vt
    
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
                    
#                     # Gradient clipping to prevent explosion
#                     torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    
                    optimizer.step()
                
                    # Save best model
                    if loss.item() < self.best_loss:
                        self.best_loss = loss.item()
                        self.best_model_state = {k: v.clone() for k, v in self.model.state_dict().items()}
                else:
                    logging.info(f"Skipping step {i} due to invalid loss: {loss.item()}")
                    continue
                
                # early stopping
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
                traceback.print_exc()
                continue
    
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
            axs[i].plot(lst)
            axs[i].grid()
            axs[i].title.set_text(label)
            if label == "train_loss":
                axs[i].set_yscale("log")
        plt.tight_layout()
        plt.show()

class ResidualBlock(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.activation = nn.ReLU()
        self.residual = (in_dim == out_dim)

    def forward(self, x):
        out = self.activation(self.linear(x))
        if self.residual:
            return out + x
        return out

class TimeConditionedMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super().__init__()
        self.input_dim = input_dim + 1  # +1 for time
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim

        layers = []
        prev_dim = self.input_dim
        for hidden_dim in hidden_dims:
            layers.append(ResidualBlock(prev_dim, hidden_dim))
            prev_dim = hidden_dim

        self.hidden_layers = nn.Sequential(*layers)
        self.output_layer = nn.Linear(prev_dim, output_dim)

    def forward(self, inputs, t):
        x = torch.cat([inputs, t[:, :1]], dim=1)
        x = self.hidden_layers(x)
        return self.output_layer(x)



def get_test_loader(batch_size=128):
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),  # CIFAR-10 mean
                             (0.2023, 0.1994, 0.2010))  # CIFAR-10 std
    ])

    test_set = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test
    )

    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2)
    return test_loader


def evaluate(model, test_loader, device='cuda' if torch.cuda.is_available() else 'cpu'):
    model.eval()
    model.to(device)

    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    accuracy = 100.0 * correct / total
    return accuracy

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main():
    # Set up configuration
    weight_shapes = [
        (32,3,3,3),      # conv1
        (64,32,3,3),     # conv2
        (128,64,3,3),    # conv3
        (512,128,3,3),   # head.0
        (10,512,1,1)     # head.2 (num_classes=10)
    ]
    bias_shapes = [
        (32,), (64,), (128,), (512,), (10,)
    ]
    
    batch_size = 4
    
    # Create MNIST test dataset for evaluation
    test_loader = get_test_loader()
    
  
    logging.info("Creating permuted model dataset using rebasin...")
    ref_point = 0  # Choose a reference model

    ref_model, original_models, permuted_models = get_permuted_models_data(ref_point=ref_point)

    # Create WSO objects from permuted models
    logging.info("Converting models to WeightSpaceObjects...")
    weights_list = []
    for model in tqdm(permuted_models):
        weights = [
            model.conv1.weight.data.clone(),
            model.conv2.weight.data.clone(),
            model.conv3.weight.data.clone(),
            model.head[0].weight.data.clone(),
            model.head[2].weight.data.clone()
        ]
        biases = [
            model.conv1.bias.data.clone(),
            model.conv2.bias.data.clone(),
            model.conv3.bias.data.clone(),
            model.head[0].bias.data.clone(),
            model.head[2].bias.data.clone()
        ]
        wso = WeightSpaceObject(weights, biases)
        weights_list.append(wso)
    
    logging.info(f"Created {len(weights_list)} permuted weight configurations")
    
    # Create flat vectors
    logging.info("Converting to flat tensors...")
    flat_target_weights = torch.stack([wso.flatten(device) for wso in weights_list])
    flat_dim = flat_target_weights.shape[1]
    
    # Create source distribution from random noise
    source_std = 0.001
    flat_source_weights = torch.randn(len(weights_list), flat_dim, device=device) * source_std

    mean = flat_target_weights.mean()
    std = flat_target_weights.std()
    
    flat_target_weights = (flat_target_weights - mean) / (std + 1e-20)
    flat_source_weights = (flat_source_weights - mean) / (std + 1e-20)

    # arr_target = flat_target_weights.cpu().numpy()
    # arr_source = flat_source_weights.cpu().numpy()
    
    # kpca_dim = 196
    # kpca = KernelPCA(n_components=kpca_dim, kernel='rbf', gamma=1e-3, fit_inverse_transform=True, random_state=0)
    # reduced_target = kpca.fit_transform(arr_target)
    # reduced_source = kpca.transform(arr_source)
    
    # target_dataset = TensorDataset(torch.from_numpy(reduced_target).float().to(device))
    # source_dataset = TensorDataset(torch.from_numpy(reduced_source).float().to(device))
    
    # sourceloader = DataLoader(source_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    # targetloader = DataLoader(target_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    # flow_model = TimeConditionedMLP(
    #     input_dim=kpca_dim,
    #     hidden_dims=[1024, 256, 64],
    #     output_dim=kpca_dim,
    # ).to(device)

    
    # Create datasets and loaders with drop_last=True to ensure consistent batch sizes
    source_dataset = TensorDataset(flat_source_weights)
    target_dataset = TensorDataset(flat_target_weights)

    sourceloader = DataLoader(source_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    targetloader = DataLoader(target_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
            
    # Create flow model
    flow_model = model = TimeConditionedMLP(
        input_dim=flat_dim,
        hidden_dims=[1024, 256, 64],
        output_dim=flat_dim,
    ).to(device)

    
    # # Set to training mode
    flow_model.train()
    
    # Count parameters
    n_params_base = sum(p.numel() for p in ConvNet().parameters())

    n_params_flow = count_parameters(flow_model)
    logging.info(f"ConvNet params:{n_params_base}")
    logging.info(f"Flow model params:{n_params_flow}")
    
    optimizer = torch.optim.Adam(flow_model.parameters(), lr=0.001)

    # Create and train the OT-CFM model
    cfm = SimpleCFM(
        sourceloader=sourceloader,
        targetloader=targetloader,
        model=flow_model,
        fm_type="ot",
        mode="ot",
        t_dist="beta",
        device=device,
        normalize_pred=True,
        geometric=True,
    )

    
    logging.info("Training flow model...")
    cfm.train(
        n_iters=50000,
        optimizer=optimizer,
        sigma=0.001,
        patience=20,
        log_freq=2,
    )
    
    # Plot training metrics
    cfm.plot_metrics()
    
    logging.info("Generating new weights...")
    n_samples = 100
    
    # random_flat = torch.randn(n_samples, kpca_dim, device=device) * source_std
    random_flat = torch.randn(n_samples, flat_dim, device=device) * source_std
    
    new_weights_flat = cfm.map(
        random_flat, 
        n_steps=100,
        noise_scale=0.0005
    )

    for i in range(n_samples):

        # reconstructed = kpca.inverse_transform(new_weights_flat[i].cpu().numpy().reshape(1, -1))
        # reconstructed_tensor = torch.from_numpy(reconstructed).float().to(device).flatten()
        # print(reconstructed_tensor.shape)
        # new_wso = WeightSpaceObject.from_flat(
        #     reconstructed_tensor,
        #     weight_shapes,
        #     bias_shapes,
        #     device=device
        # )
        
        new_wso = WeightSpaceObject.from_flat(
            new_weights_flat[i],
            weight_shapes,
            bias_shapes,
            device=device
        )


        # Create and test model
        model = ConvNet()
        model.conv1.weight.data = new_wso.weights[0].clone()
        model.conv2.weight.data = new_wso.weights[1].clone()
        model.conv3.weight.data = new_wso.weights[2].clone()
        model.head[0].weight.data = new_wso.weights[3].clone()
        model.head[2].weight.data = new_wso.weights[4].clone()
        model.conv1.bias.data = new_wso.biases[0].clone()
        model.conv2.bias.data = new_wso.biases[1].clone()
        model.conv3.bias.data = new_wso.biases[2].clone()
        model.head[0].bias.data = new_wso.biases[3].clone()
        model.head[2].bias.data = new_wso.biases[4].clone()

        acc = evaluate(model, test_loader)
        logging.info(f"Generated model {i} accuracy: {acc:.2f}%")
    


if __name__ == "__main__":
    main()