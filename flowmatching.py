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



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.basicConfig(stream=sys.stdout, format='%(asctime)s %(levelname)s: %(message)s', level=logging.INFO, datefmt='%I:%M:%S')

# PermutationSpec class similar to the JAX version but using PyTorch
class PermutationSpec(NamedTuple):
    perm_to_axes: dict
    axes_to_perm: dict

def permutation_spec_from_axes_to_perm(axes_to_perm: dict) -> PermutationSpec:
    perm_to_axes = defaultdict(list)
    for wk, axis_perms in axes_to_perm.items():
        for axis, perm in enumerate(axis_perms):
            if perm is not None:
                perm_to_axes[perm].append((wk, axis))
    return PermutationSpec(perm_to_axes=dict(perm_to_axes), axes_to_perm=axes_to_perm)

def mlp_permutation_spec_mlp() -> PermutationSpec:
    """Define permutation spec for MLP architecture"""
    return permutation_spec_from_axes_to_perm({
        "fc1.weight": (None, "P_0"),       # Input (None) to fc1 output (P_0)
        "fc1.bias": ("P_0",),              # Bias for fc1 output (P_0)
        "fc2.weight": ("P_0", "P_1"),      # fc1 output (P_0) to fc2 output (P_1)
        "fc2.bias": ("P_1",),              # Bias for fc2 output (P_1)
        "fc3.weight": ("P_1", None),       # fc2 output (P_1) to fc3 output (None)
        "fc3.bias": (None,),               # Bias for fc3 output (None)
    })

def get_permuted_param(ps: PermutationSpec, perm, k: str, params, except_axis=None):
    """Get parameter k from params, with permutations applied."""
    w = params[k]
    for axis, p in enumerate(ps.axes_to_perm[k]):
        # Skip the axis we're trying to permute
        if axis == except_axis:
            continue

        # None indicates no permutation for that axis
        if p is not None:
            w = torch.index_select(w, axis, torch.tensor(perm[p], device=w.device))

    return w

def apply_permutation(ps: PermutationSpec, perm, params):
    """Apply permutation to params"""
    return {k: get_permuted_param(ps, perm, k, params) for k in params.keys()}

def weight_matching(ps: PermutationSpec, params_a, params_b, max_iter=100, init_perm=None, silent=True, device=None):
    """Find permutation of params_b to make them match params_a."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Move all tensors to the correct device
    params_a = {k: v.to(device) for k, v in params_a.items()}
    params_b = {k: v.to(device) for k, v in params_b.items()}

    # Get permutation sizes from the first parameter with each permutation
    perm_sizes = {p: params_a[axes[0][0]].shape[axes[0][1]] 
                  for p, axes in ps.perm_to_axes.items()}
    
    # Initialize permutations to identity if none provided
    if init_perm is None:
        perm = {p: torch.arange(n, device=device) for p, n in perm_sizes.items()}
    else:
        perm = {p: v.to(device) for p, v in init_perm.items()}
        
    perm_names = list(perm.keys())
    
    # Use a random number generator with a fixed seed for reproducibility
    rng = np.random.RandomState(42)

    for iteration in range(max_iter):
        progress = False
        
        # Shuffle the order of permutations to update
        for p_ix in rng.permutation(len(perm_names)):
            p = perm_names[p_ix]
            n = perm_sizes[p]
            
            # Initialize cost matrix
            A = torch.zeros((n, n), device=device)
            
            # Fill in cost matrix based on all parameters affected by this permutation
            for wk, axis in ps.perm_to_axes[p]:
                w_a = params_a[wk]
                w_b = get_permuted_param(ps, perm, wk, params_b, except_axis=axis)

                w_a = w_a.moveaxis(axis, 0).reshape((n, -1))
                w_b = w_b.moveaxis(axis, 0).reshape((n, -1))

                A += w_a @ w_b.T

            # Solve the linear assignment problem
            ri, ci = linear_sum_assignment(A.detach().cpu().numpy(), maximize=True)
            assert (ri == np.arange(len(ri))).all()

            # Calculate improvement
            eye_old = torch.eye(n, device=device)[perm[p]]
            eye_new = torch.eye(n, device=device)[ci]

            oldL = torch.tensordot(A, eye_old, dims=([0, 1], [0, 1]))
            newL = torch.tensordot(A, eye_new, dims=([0, 1], [0, 1]))

            if not silent and newL > oldL + 1e-12:
                logging.info(f"{iteration}/{p}: {newL.item() - oldL.item()}")

            progress = progress or newL > oldL + 1e-12

            perm[p] = torch.tensor(ci, device=device)

        if not progress:
            break

    return perm


def update_model_weights(model, aligned_params):
    """Update model weights with aligned parameters"""
    # Convert numpy arrays to torch tensors if needed
    model.fc1.weight.data = aligned_params["fc1.weight"].T
    model.fc1.bias.data = aligned_params["fc1.bias"]
    model.fc2.weight.data = aligned_params["fc2.weight"].T
    model.fc2.bias.data = aligned_params["fc2.bias"]
    model.fc3.weight.data = aligned_params["fc3.weight"].T
    model.fc3.bias.data = aligned_params["fc3.bias"]
    
def load_model_weights(model, model_path):
    """Load model weights from file"""
    weights, biases = torch.load(model_path, map_location=device)
    model.fc1.weight.data = weights[0]
    model.fc1.bias.data = biases[0]
    model.fc2.weight.data = weights[1]
    model.fc2.bias.data = biases[1]
    model.fc3.weight.data = weights[2]
    model.fc3.bias.data = biases[2]
    return model.to(device)

def get_permuted_models_data(ref_point=0, model_dir="../models", num_models=251):
    """Apply weight matching to align models with a reference model"""
    # Create reference model
    ref_model = MLP()  # Assumes MLP class is defined
    ref_model_path = f"{model_dir}/mlp_weights_{ref_point}.pt"
    ref_model = load_model_weights(ref_model, ref_model_path).to(device)
    
    ps = mlp_permutation_spec_mlp()
    
    # Convert reference model weights to dictionary format
    params_a = {
        "fc1.weight": ref_model.fc1.weight.T.to(device),
        "fc1.bias": ref_model.fc1.bias.to(device),
        "fc2.weight": ref_model.fc2.weight.T.to(device),
        "fc2.bias": ref_model.fc2.bias.to(device),
        "fc3.weight": ref_model.fc3.weight.T.to(device),
        "fc3.bias": ref_model.fc3.bias.to(device),
    }
    
    org_models = []
    permuted_models = []

    for i in range(0, num_models):
        if i == ref_point:
            continue
            
        model_path = f"{model_dir}/mlp_weights_{i}.pt"

        model = MLP()  # Assumes MLP class is defined
        model = load_model_weights(model, model_path).to(device)
        org_models.append(model)
        
        # Convert model weights to dictionary format
        params_b = {
                "fc1.weight": model.fc1.weight.T.to(device),
                "fc1.bias": model.fc1.bias.to(device),
                "fc2.weight": model.fc2.weight.T.to(device),
                "fc2.bias": model.fc2.bias.to(device),
                "fc3.weight": model.fc3.weight.T.to(device),
                "fc3.bias": model.fc3.bias.to(device),
        }

        # Find permutation to align with reference model
        perm = weight_matching(ps, params_a, params_b)
        
        # Apply permutation to model_b
        aligned_params_b = apply_permutation(ps, perm, params_b)
        
        # Create a new model with permuted weights
        reconstructed_model = copy.deepcopy(model)
        update_model_weights(reconstructed_model, aligned_params_b)
        
        permuted_models.append(reconstructed_model.to(device))

            
    return ref_model, org_models, permuted_models


import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.nn.functional as F
import copy
from collections import defaultdict
import os
import traceback

# MLP model definition
class MLP(nn.Module):
    def __init__(self, init_type='xavier', seed=None):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(784, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, 10)
        
        if seed is not None:
            torch.manual_seed(seed)

        self.init_weights(init_type)

    def init_weights(self, init_type):
        if init_type == 'xavier':
            nn.init.xavier_uniform_(self.fc1.weight)
            nn.init.xavier_uniform_(self.fc2.weight)
            nn.init.xavier_uniform_(self.fc3.weight)
        elif init_type == 'he':
            nn.init.kaiming_uniform_(self.fc1.weight, nonlinearity='relu')
            nn.init.kaiming_uniform_(self.fc2.weight, nonlinearity='relu')
            nn.init.kaiming_uniform_(self.fc3.weight, nonlinearity='relu')
        else:
            nn.init.normal_(self.fc1.weight)
            nn.init.normal_(self.fc2.weight)
            nn.init.normal_(self.fc3.weight)
        
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)
        nn.init.zeros_(self.fc3.bias)

    def forward(self, x):
        x = x.view(-1, 784)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# WeightSpaceObject class for handling MLP weights
class WeightSpaceObject:
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
        layer_layout,
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
        self.layer_layout = layer_layout
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


def test_mlp(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    model = model.to(device)
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    return 100 * correct / total

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# class SafeRelationalTransformer(nn.Module):
#     def __init__(self, original_transformer):
#         super().__init__()
#         self.transformer = original_transformer
        
#     def forward(self, xt, t):
#         """Safe forward pass that handles batch size mismatches"""

#         # Get expected batch size from the transformer
#         expected_batch_size = self.transformer.n_batch
#         actual_batch_size = xt.size(0)
        
#         # If batch sizes don't match, adjust the input
#         if actual_batch_size != expected_batch_size:
#             # Either pad or truncate to match expected size
#             if actual_batch_size < expected_batch_size:
#                 # Pad by repeating the last sample
#                 padding = xt[-1].unsqueeze(0).repeat(expected_batch_size - actual_batch_size, 1)
#                 xt_adjusted = torch.cat([xt, padding], dim=0)
#                 t_adjusted = torch.cat([t, t[-1].unsqueeze(0).repeat(expected_batch_size - actual_batch_size, 1)], dim=0)
#             else:
#                 # Truncate
#                 xt_adjusted = xt[:expected_batch_size]
#                 t_adjusted = t[:expected_batch_size]
            
#             # Forward pass with adjusted batch
#             output = self.transformer(xt_adjusted, t_adjusted)
            
#             # Return only the valid part if we padded
#             if actual_batch_size < expected_batch_size:
#                 return output[:actual_batch_size]
#             else:
#                 return output
#         else:
#             # Normal forward pass if batch sizes match
#             return self.transformer(xt, t)
                

def main():
    # Set up configuration
    layer_layout = [784, 32, 32, 10]  # MLP architecture for MNIST
    batch_size = 4
    
    # Create MNIST test dataset for evaluation
    from torchvision import datasets, transforms
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    test_data = datasets.MNIST('.', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=32, shuffle=False)
    
  
    logging.info("Creating permuted model dataset using rebasin...")
    ref_point = 0  # Choose a reference model

    ref_model, original_models, permuted_models = get_permuted_models_data(ref_point=ref_point)

    
    # Create WSO objects from permuted models
    logging.info("Converting models to WeightSpaceObjects...")
    weights_list = []
    for model in tqdm(permuted_models):
        weights = (
            model.fc1.weight.data.clone(),
            model.fc2.weight.data.clone(),
            model.fc3.weight.data.clone()
        )
        
        biases = (
            model.fc1.bias.data.clone(),
            model.fc2.bias.data.clone(), 
            model.fc3.bias.data.clone()
        )
        
        wso = WeightSpaceObject(weights, biases)
        weights_list.append(wso)
    
    logging.info(f"Created {len(weights_list)} permuted weight configurations")
    
    # Create flat vectors
    logging.info("Converting to flat tensors...")
    flat_target_weights = torch.stack([wso.flatten(device) for wso in weights_list])
    flat_dim = flat_target_weights.shape[1]
    
    # Create source distribution from random noise
    source_std = 0.01
    flat_source_weights = torch.randn(len(weights_list), flat_dim, device=device) * source_std
    
    # Create datasets and loaders with drop_last=True to ensure consistent batch sizes
    source_dataset = TensorDataset(flat_source_weights)
    target_dataset = TensorDataset(flat_target_weights)
    
    sourceloader = DataLoader(source_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    targetloader = DataLoader(target_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    
    flow_model = model = TimeConditionedMLP(
        input_dim=flat_dim,
        hidden_dims=[2048, 1024, 512, 128, 64],
        output_dim=flat_dim,
    ).to(device)    
    
    # # Set to training mode
    flow_model.train()
    
    # Count parameters
    n_params_base = sum(p.numel() for p in MLP().parameters())
    n_params_flow = count_parameters(flow_model)
    logging.info(f"MLP params:{n_params_base}")
    logging.info(f"Flow model params:{n_params_flow}")
    
    # Create optimizer with gradient clipping
    optimizer = torch.optim.Adam(flow_model.parameters(), lr=0.001)
    
    # Create and train the CFM model
    cfm = SimpleCFM(
        sourceloader=sourceloader,
        targetloader=targetloader,
        model=flow_model,
        layer_layout=np.array(layer_layout),
        fm_type="ot",  # or "ot" for optimal transport
        mode="ot",
        t_dist="beta",
        device=device,
        normalize_pred=True,
        geometric=True,
    )
    
    logging.info("Training flow model...")
    cfm.train(
        n_iters=20000,
        optimizer=optimizer,
        sigma=0.001,
        patience=20,
        log_freq=2,
    )
    
    # Plot training metrics
    cfm.plot_metrics()
    
    # Generate new MLP weights
    logging.info("Generating new MLP weights...")
    n_samples = 100
    random_flat = torch.randn(n_samples, flat_dim, device=device) * source_std
    
    new_weights_flat = cfm.map(
        random_flat, 
        n_steps=100,
        noise_scale=0.0005
    )

    # Convert to MLP weights and save
    for i in range(n_samples):
        new_wso = WeightSpaceObject.from_flat(
            new_weights_flat[i], 
            layers=np.array(layer_layout), 
            device=device
        )

        expected_weight_shapes = [(32, 784), (32, 32), (10, 32)]
        expected_bias_shapes = [(32,), (32,), (10,)]

        assert len(new_wso.weights) == 3, f"Expected 3 weight matrices, got {len(new_wso.weights)}"
        assert len(new_wso.biases) == 3, f"Expected 3 bias vectors, got {len(new_wso.biases)}"
        
        # Check each weight and bias shape
        for j, (w, expected_shape) in enumerate(zip(new_wso.weights, expected_weight_shapes)):
            assert w.shape == expected_shape, f"Weight {j} has shape {w.shape}, expected {expected_shape}"
        
        for j, (b, expected_shape) in enumerate(zip(new_wso.biases, expected_bias_shapes)):
            assert b.shape == expected_shape, f"Bias {j} has shape {b.shape}, expected {expected_shape}"

        # Save the generated weights
        torch.save(
            (new_wso.weights, new_wso.biases),
            f"generated_mlp_weights_{i}.pt"
        )

        # Create and test model
        model = MLP()
        model.fc1.weight.data = new_wso.weights[0].clone()
        model.fc1.bias.data = new_wso.biases[0].clone()
        model.fc2.weight.data = new_wso.weights[1].clone()
        model.fc2.bias.data = new_wso.biases[1].clone()
        model.fc3.weight.data = new_wso.weights[2].clone()
        model.fc3.bias.data = new_wso.biases[2].clone()

        acc = test_mlp(model, test_loader)
        logging.info(f"Generated model {i} accuracy: {acc:.2f}%")
    


if __name__ == "__main__":
    main()