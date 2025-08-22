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
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from torch.optim import Adam, SGD
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torchvision import transforms, datasets
import torchvision
from collections import defaultdict
import os
import traceback

from sklearn.neighbors import KernelDensity
from sklearn.cluster import MeanShift
from scipy.stats import gaussian_kde

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.basicConfig(stream=sys.stdout, format='%(asctime)s %(levelname)s: %(message)s', level=logging.INFO, datefmt='%I:%M:%S')

class Bunch:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super().__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear = nn.Linear(64, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))  # 32x32
        out = self.layer1(out)  # 32x32
        out = self.layer2(out)  # 16x16
        out = self.layer3(out)  # 8x8
        out = F.avg_pool2d(out, 8)  # Global avg pool
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def ResNet20():
    return ResNet(BasicBlock, [3,3,3])

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

def resnet_permutation_spec() -> PermutationSpec:
    conv = lambda name, p_in, p_out: {f"{name}.weight": (p_out, p_in, None, None)}

    norm = lambda name, p: {
        f"{name}.weight": (p,),
        f"{name}.bias": (p,),
        f"{name}.running_mean": (p,),
        f"{name}.running_var": (p,),
    }

    dense = lambda name, p_in, p_out: {
        f"{name}.weight": (p_out, p_in),
        f"{name}.bias": (p_out,)
    }

    easyblock = lambda name, p: {
        **conv(f"{name}.conv1", p, f"P_{name}_inner"),
        **norm(f"{name}.bn1", f"P_{name}_inner"),
        **conv(f"{name}.conv2", f"P_{name}_inner", p),
        **norm(f"{name}.bn2", p),
    }

    shortcutblock = lambda name, p_in, p_out: {
        **conv(f"{name}.conv1", p_in, f"P_{name}_inner"),
        **norm(f"{name}.bn1", f"P_{name}_inner"),
        **conv(f"{name}.conv2", f"P_{name}_inner", p_out),
        **norm(f"{name}.bn2", p_out),
        **conv(f"{name}.shortcut.0", p_in, p_out),
        **norm(f"{name}.shortcut.1", p_out),
    }

    return permutation_spec_from_axes_to_perm({
        **conv("conv1", None, "P_bg0"),
        **norm("bn1", "P_bg0"),

        **easyblock("layer1.0", "P_bg0"),
        **easyblock("layer1.1", "P_bg0"),
        **easyblock("layer1.2", "P_bg0"),

        **shortcutblock("layer2.0", "P_bg0", "P_bg1"),
        **easyblock("layer2.1", "P_bg1"),
        **easyblock("layer2.2", "P_bg1"),

        **shortcutblock("layer3.0", "P_bg1", "P_bg2"),
        **easyblock("layer3.1", "P_bg2"),
        **easyblock("layer3.2", "P_bg2"),

        **dense("linear", "P_bg2", None),
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

    params_a = {k: v.to(device) for k, v in params_a.items()}
    params_b = {k: v.to(device) for k, v in params_b.items()}

    perm_sizes = {p: params_a[axes[0][0]].shape[axes[0][1]] 
                  for p, axes in ps.perm_to_axes.items()}
    
    if init_perm is None:
        perm = {p: torch.arange(n, device=device) for p, n in perm_sizes.items()}
    else:
        perm = {p: v.to(device) for p, v in init_perm.items()}
        
    perm_names = list(perm.keys())
    
    rng = np.random.RandomState(42)

    for iteration in range(max_iter):
        progress = False
        
        for p_ix in rng.permutation(len(perm_names)):
            p = perm_names[p_ix]
            n = perm_sizes[p]
            
            A = torch.zeros((n, n), device=device)
            
            for wk, axis in ps.perm_to_axes[p]:
                w_a = params_a[wk]
                w_b = get_permuted_param(ps, perm, wk, params_b, except_axis=axis)

                w_a = w_a.moveaxis(axis, 0).reshape((n, -1))
                w_b = w_b.moveaxis(axis, 0).reshape((n, -1))

                A += w_a @ w_b.T

            ri, ci = linear_sum_assignment(A.detach().cpu().numpy(), maximize=True)
            assert (ri == np.arange(len(ri))).all()

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

def get_permuted_models_data(ref_point=0, model_dir="imagenet_resnet_models", num_models=250, device=None):
    """Apply weight matching to align models with a reference model"""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    ref_model = ResNet20()
    ref_model_path = f"{model_dir}/resnet_weights_{ref_point}.pt"
    
    try:
        ref_model.load_state_dict(torch.load(ref_model_path, map_location=device))
        ref_model = ref_model.to(device)
        logging.info(f"Loaded reference model from {ref_model_path}")
    except Exception as e:
        logging.error(f"Failed to load reference model: {e}")
        raise e
    
    ps = resnet_permutation_spec()
    
    params_a = {k: v.clone().detach() for k, v in ref_model.state_dict().items() 
               if k in ps.axes_to_perm}
    
    permuted_models = []

    for i in tqdm(range(num_models), desc="Processing models"):
        if i == ref_point:
            continue
        
        model_path = f"{model_dir}/resnet_weights_{i}.pt"
        if not os.path.exists(model_path):
            logging.info(f"Skipping model {i} - file not found")
            continue
        
        try:
            # Load model B
            model_b = ResNet20()
            model_b.load_state_dict(torch.load(model_path, map_location=device))
            model_b = model_b.to(device)

            # Extract params and buffers
            params_b = {k: v.clone().detach() for k, v in model_b.state_dict().items() 
                       if k in ps.axes_to_perm}
            
            # Perform weight matching directly in PyTorch
            perm = weight_matching(ps, params_a, params_b, device=device)
            
            # Apply permutation
            permuted_params_b = apply_permutation(ps, perm, params_b)
            
            reconstructed_model = copy.deepcopy(model_b)
            state_dict = reconstructed_model.state_dict()
            
            for k in permuted_params_b:
                state_dict[k] = permuted_params_b[k]
            
            reconstructed_model.load_state_dict(state_dict)
            reconstructed_model = reconstructed_model.to(device)
            
            # # Evaluate accuracy before adding to list
            # test_loader = get_test_loader()
            # accuracy = evaluate(reconstructed_model, test_loader)
            # logging.info(f"Model {i} accuracy after matching: {accuracy:.2f}%")
            
            permuted_models.append(reconstructed_model)
        
        except Exception as e:
            logging.error(f"Error processing model {i}: {e}")
            continue
        
        torch.cuda.empty_cache()
    
    logging.info(f"Processed {len(permuted_models)} models successfully")
    return ref_model, permuted_models

class WeightSpaceObject:
    def __init__(self, weights, biases):
        self.weights = tuple(weights)
        self.biases = tuple(biases)
        
    def flatten(self, device=None):
        flat = torch.cat([w.flatten() for w in self.weights] + 
                        [b.flatten() for b in self.biases])
        return flat.to(device) if device else flat
    
    @classmethod
    def from_flat(cls, flat, weight_shapes, bias_shapes, device=None):
        sizes = ([np.prod(s) for s in weight_shapes + bias_shapes])
        parts = []
        start = 0
        for size in sizes:
            parts.append(flat[start:start+size])
            start += size
            
        n_weights = len(weight_shapes)
        n_biases = len(bias_shapes)
        
        weights = [parts[i].reshape(weight_shapes[i]) for i in range(n_weights)]
        biases = [parts[n_weights + i].reshape(bias_shapes[i]) for i in range(n_biases)]        
        return cls(weights, biases)
        
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

class SimpleCFM:
    def __init__(
        self,
        sourceloader,
        targetloader,
        model,
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
        self.mode = mode
        self.sigma = 0.001
        self.normalize_pred = normalize_pred
        self.geometric = geometric
        self.metrics = {"train_loss": [], "time": [], "grad_norm": [], "flow_norm": [], "true_norm": []}
        self.best_loss = float('inf')
        self.best_model_state = None
        self.input_dim = model.input_dim - 1  

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

    def sample_time_and_noisy_data(self):
        x0 = self.sample_from_loader(self.sourceloader).to(self.device)
        batch_size = x0.size(0)
    
        # Sample times t uniformly in [0,1]
        t = torch.rand(batch_size, device=self.device)
    
        # Define noise scale function sigma(t) (e.g., linear or sqrt schedule)
        sigma = 0.01 + 0.1 * t  # example, adjust for your use case
        sigma = sigma.view(-1, *([1] * (x0.dim() - 1)))  # broadcast dims
    
        noise = torch.randn_like(x0)
        xt = x0 + noise * sigma
    
        return Bunch(t=t.unsqueeze(-1), x0=x0, xt=xt, noise=noise, sigma=sigma, batch_size=batch_size)

    def forward(self, flow):
        """Forward pass through the model with proper error handling"""
        try:
            flow_pred = self.model(flow.xt, flow.t)
            return None, flow_pred
        except Exception as e:
            logging.info(f"Error in forward pass: {str(e)}")
            traceback.print_exc()
            return None, torch.zeros_like(flow.ut)
        
    def loss_fn(self, flow_pred, flow):
        """Compute loss between predicted and true flows"""
        if self.mode == "target":
            l_flow = torch.mean((flow_pred.squeeze() - flow.x1) ** 2)
        elif self.mode == "velocity":
            l_flow = torch.mean((flow_pred.squeeze() - flow.ut) ** 2)
        else:
            l_flow = torch.mean((flow_pred.squeeze() - flow.ut) ** 2)
            
        return None, l_flow
    
    def map(self, x0, n_steps=20, return_traj=False, noise_scale=0.001):
        """Map points using the flow model to generate new weights"""
        if self.best_model_state is not None:
            current_state = {k: v.clone() for k, v in self.model.state_dict().items()}
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

        if return_traj:
            traj.append(xt.detach().clone())

        if self.best_model_state is not None:
            self.model.load_state_dict(current_state)
            
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
                    
#                   # Gradient clipping to prevent explosion
                    # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    
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


class EnhancedWeightSpaceCFM(SimpleCFM):
    def __init__(self, sourceloader, targetloader, model, **kwargs):
        super().__init__(sourceloader, targetloader, model, **kwargs)
        
        # Enhanced settings for weight spaces
        self.sigma = 0.0005  # Lower noise for stable weight distributions
        self.mode = "velocity"  # Velocity parameterization works well
        
    def sample_time_and_flow(self):
        """Enhanced sampling with better time distribution for weight spaces"""
        x0 = self.sample_from_loader(self.sourceloader)
        x1 = self.sample_from_loader(self.targetloader)
        
        batch_size = min(x0.size(0), x1.size(0))
        x0 = x0[:batch_size].to(self.device)
        x1 = x1[:batch_size].to(self.device)
        

        if self.t_dist == "beta":
            alpha, beta_param = 2.0, 5.0  # Concentrate mass near t=0
            t = torch.distributions.Beta(alpha, beta_param).sample((batch_size,)).to(self.device)
        else:
            t = torch.rand(batch_size).to(self.device)
        
        t_pad = t.reshape(-1, *([1] * (x0.dim() - 1)))
        
        mu_t = (1 - t_pad) * x0 + t_pad * x1
        sigma_pad = torch.tensor(self.sigma).to(self.device)
        epsilon = torch.randn_like(x0) * sigma_pad
        xt = mu_t + epsilon
        
        ut = x1 - x0
        
        return Bunch(t=t.unsqueeze(-1), x0=x0, xt=xt, x1=x1, ut=ut, 
                    eps=epsilon, batch_size=batch_size)
        
    def map(self, x0, n_steps=50, return_traj=False, method="euler"):
        """Enhanced mapping function optimized for weight spaces"""
        
        if self.best_model_state is not None:
            current_state = {k: v.clone() for k, v in self.model.state_dict().items()}
            self.model.load_state_dict(self.best_model_state)

        self.model.eval()
        batch_size, flat_dim = x0.size()
        
        traj = [x0.detach().clone()] if return_traj else None
        xt = x0.clone()

        if method == "euler":
            times = torch.linspace(0, 1, n_steps).to(self.device)
            dt = times[1] - times[0]

            for i, t in enumerate(times[:-1]):
                with torch.no_grad():
                    t_tensor = torch.ones(batch_size, 1).to(self.device) * t
                    

                    pred = self.model(xt, t_tensor)
                    if pred.dim() > 2:
                        pred = pred.squeeze(-1)

                    # Get velocity
                    if self.mode == "velocity":
                        vt = pred
                    else:
                        vt = pred - xt

                    # Euler step
                    xt = xt + vt * dt
                
                if return_traj:
                    traj.append(xt.detach().clone())
                    
        elif method == "rk4":
            times = torch.linspace(0, 1, n_steps).to(self.device)
            dt = times[1] - times[0]
            
            for i, t in enumerate(times[:-1]):
                with torch.no_grad():
                    t_tensor = torch.ones(batch_size, 1).to(self.device) * t
                    
                    # k1
                    pred1 = self.model(xt, t_tensor)
                    if pred1.dim() > 2:
                        pred1 = pred1.squeeze(-1)
                    k1 = pred1 if self.mode == "velocity" else pred1 - xt
                    
                    # k2  
                    xt_k2 = xt + 0.5 * dt * k1
                    t_k2 = t_tensor + 0.5 * dt
                    pred2 = self.model(xt_k2, t_k2)
                    if pred2.dim() > 2:
                        pred2 = pred2.squeeze(-1)
                    k2 = pred2 if self.mode == "velocity" else pred2 - xt_k2
                    
                    # k3
                    xt_k3 = xt + 0.5 * dt * k2  
                    pred3 = self.model(xt_k3, t_k2)
                    if pred3.dim() > 2:
                        pred3 = pred3.squeeze(-1)
                    k3 = pred3 if self.mode == "velocity" else pred3 - xt_k3
                    
                    # k4
                    xt_k4 = xt + dt * k3
                    t_k4 = t_tensor + dt
                    pred4 = self.model(xt_k4, t_k4)
                    if pred4.dim() > 2:
                        pred4 = pred4.squeeze(-1)
                    k4 = pred4 if self.mode == "velocity" else pred4 - xt_k4
                    
                    # Final RK4 step
                    xt = xt + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
                        
                if return_traj:
                    traj.append(xt.detach().clone())

        if self.best_model_state is not None:
            self.model.load_state_dict(current_state)
        self.model.train()

        return traj if return_traj else xt

    def generate_weights(self, n_samples=10, source_noise_std=0.001, **map_kwargs):
        """Generate new weight configurations"""
        
        flat_dim = self.input_dim
        source_samples = torch.randn(n_samples, flat_dim, device=self.device) * source_noise_std
        
        generated_weights = self.map(source_samples, **map_kwargs)
        
        return generated_weights
    
    def interpolate_weights(self, weight1, weight2, n_steps=10):
        """Interpolate between two weight configurations through the flow"""
        
        alphas = torch.linspace(0, 1, n_steps).to(self.device)
        interpolated = []
        
        for alpha in alphas:
            x_interp = (1 - alpha) * weight1 + alpha * weight2
            
            mapped = self.map(x_interp.unsqueeze(0), n_steps=20)
            interpolated.append(mapped.squeeze(0))
            
        return torch.stack(interpolated)

    
class WeightSpaceFlowModel(nn.Module):
    def __init__(self, input_dim, time_embed_dim=64):
        super().__init__()
        self.time_embed_dim = time_embed_dim
        self.input_dim = input_dim
        
        self.time_embed = nn.Sequential(
            nn.Linear(1, time_embed_dim),
            nn.GELU(),
            nn.Linear(time_embed_dim, time_embed_dim)
        )
        
        hidden_dim = min(1024, input_dim // 4)
        
        self.net = nn.Sequential(
            nn.Linear(input_dim + time_embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim), 
            nn.GELU(),
            nn.Dropout(0.1),
            
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            
            nn.Linear(hidden_dim // 2, input_dim)
        )
        
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
    
    def forward(self, x, t):
        t_embed = self.time_embed(t)
        
        combined = torch.cat([x, t_embed], dim=-1)
        
        return self.net(combined)

# Training recommendations
def train_weight_space_flow(sourceloader, targetloader, input_dim):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    flow_model = WeightSpaceFlowModel(input_dim).to(device)
    
    cfm = EnhancedWeightSpaceCFM(
        sourceloader=sourceloader,
        targetloader=targetloader,
        model=flow_model,
        mode="velocity",
        t_dist="beta", 
        device=device
    )
    
    optimizer = torch.optim.AdamW(
        flow_model.parameters(), 
        lr=1e-4,
        weight_decay=1e-5,
        betas=(0.9, 0.95)
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=20000, eta_min=1e-6
    )
    
    cfm.train(
        n_iters=20000,
        optimizer=optimizer,
        sigma=0.001, 
        patience=100, 
        log_freq=10
    )
    
    return cfm

def recalibrate_bn_stats(model, calibration_loader, device='cuda', print_stats=True):
    """Recalculate BatchNorm statistics for generated weights"""
    model.train()
    model.to(device)
    
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.reset_running_stats()
            m.momentum = None
    
    with torch.no_grad():
        for inputs, _ in calibration_loader:
            inputs = inputs.to(device)
            _ = model(inputs)
    
    if print_stats:
        for name, m in model.named_modules():
            if isinstance(m, nn.BatchNorm2d):
                logging.info(f"{name}: mean={m.running_mean.mean().item():.4f}, "
                      f"var={m.running_var.mean().item():.4f}, "
                      f"num_batches_tracked={m.num_batches_tracked.item()}")
    

    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.momentum = 0.1
    
    return model

 
def main():
    weight_shapes = [
        (16, 3, 3, 3),
        (16,),
        (16, 16, 3, 3), (16,),
        (16, 16, 3, 3), (16,),
        (16, 16, 3, 3), (16,),
        (16, 16, 3, 3), (16,),
        (16, 16, 3, 3), (16,),
        (16, 16, 3, 3), (16,),
        (32, 16, 1, 1), (32,),
        (32, 16, 3, 3), (32,),
        (32, 32, 3, 3), (32,),
        (32, 32, 3, 3), (32,),
        (32, 32, 3, 3), (32,),
        (32, 32, 3, 3), (32,),
        (32, 32, 3, 3), (32,),
        (64, 32, 1, 1), (64,),
        (64, 32, 3, 3), (64,),
        (64, 64, 3, 3), (64,),
        (64, 64, 3, 3), (64,),
        (64, 64, 3, 3), (64,),
        (64, 64, 3, 3), (64,),
        (64, 64, 3, 3), (64,),
        (10, 64)
    ]
    
    bias_shapes = [
        (16,),
        (16,), (16,),
        (16,), (16,),
        (16,), (16,),
        (32,),
        (32,), (32,),
        (32,), (32,),
        (32,), (32,),
        (64,),
        (64,), (64,),
        (64,), (64,),
        (64,), (64,),
        (64,), (64,),
        (10,)
    ]
    
    batch_size = 4
    test_loader = get_test_loader()
  
    logging.info("Creating permuted model dataset using rebasin...")
    ref_point = 0
    ref_model, permuted_models = get_permuted_models_data(ref_point=ref_point)

    logging.info("Converting models to WeightSpaceObjects...")
    weights_list = []
    bn_stats_list = []
    
    for model in tqdm(permuted_models):
        weights = (
            model.conv1.weight.data.clone(),
            model.bn1.weight.data.clone(),
            model.layer1[0].conv1.weight.data.clone(),
            model.layer1[0].bn1.weight.data.clone(),
            model.layer1[0].conv2.weight.data.clone(),
            model.layer1[0].bn2.weight.data.clone(),
            model.layer1[1].conv1.weight.data.clone(),
            model.layer1[1].bn1.weight.data.clone(),
            model.layer1[1].conv2.weight.data.clone(),
            model.layer1[1].bn2.weight.data.clone(),
            model.layer1[2].conv1.weight.data.clone(),
            model.layer1[2].bn1.weight.data.clone(),
            model.layer1[2].conv2.weight.data.clone(),
            model.layer1[2].bn2.weight.data.clone(),
            model.layer2[0].shortcut[0].weight.data.clone(),
            model.layer2[0].shortcut[1].weight.data.clone(),
            model.layer2[0].conv1.weight.data.clone(),
            model.layer2[0].bn1.weight.data.clone(),
            model.layer2[0].conv2.weight.data.clone(),
            model.layer2[0].bn2.weight.data.clone(),
            model.layer2[1].conv1.weight.data.clone(),
            model.layer2[1].bn1.weight.data.clone(),
            model.layer2[1].conv2.weight.data.clone(),
            model.layer2[1].bn2.weight.data.clone(),
            model.layer2[2].conv1.weight.data.clone(),
            model.layer2[2].bn1.weight.data.clone(),
            model.layer2[2].conv2.weight.data.clone(),
            model.layer2[2].bn2.weight.data.clone(),
            model.layer3[0].shortcut[0].weight.data.clone(),
            model.layer3[0].shortcut[1].weight.data.clone(),
            model.layer3[0].conv1.weight.data.clone(),
            model.layer3[0].bn1.weight.data.clone(),
            model.layer3[0].conv2.weight.data.clone(),
            model.layer3[0].bn2.weight.data.clone(),
            model.layer3[1].conv1.weight.data.clone(),
            model.layer3[1].bn1.weight.data.clone(),
            model.layer3[1].conv2.weight.data.clone(),
            model.layer3[1].bn2.weight.data.clone(),
            model.layer3[2].conv1.weight.data.clone(),
            model.layer3[2].bn1.weight.data.clone(),
            model.layer3[2].conv2.weight.data.clone(),
            model.layer3[2].bn2.weight.data.clone(),
            model.linear.weight.data.clone()
        )
        
        biases = (
            model.bn1.bias.data.clone(),
            model.layer1[0].bn1.bias.data.clone(),
            model.layer1[0].bn2.bias.data.clone(),
            model.layer1[1].bn1.bias.data.clone(),
            model.layer1[1].bn2.bias.data.clone(),
            model.layer1[2].bn1.bias.data.clone(),
            model.layer1[2].bn2.bias.data.clone(),
            model.layer2[0].shortcut[1].bias.data.clone(),
            model.layer2[0].bn1.bias.data.clone(),
            model.layer2[0].bn2.bias.data.clone(),
            model.layer2[1].bn1.bias.data.clone(),
            model.layer2[1].bn2.bias.data.clone(),
            model.layer2[2].bn1.bias.data.clone(),
            model.layer2[2].bn2.bias.data.clone(),
            model.layer3[0].shortcut[1].bias.data.clone(),
            model.layer3[0].bn1.bias.data.clone(),
            model.layer3[0].bn2.bias.data.clone(),
            model.layer3[1].bn1.bias.data.clone(),
            model.layer3[1].bn2.bias.data.clone(),
            model.layer3[2].bn1.bias.data.clone(),
            model.layer3[2].bn2.bias.data.clone(),
            model.linear.bias.data.clone()
        )

        wso = WeightSpaceObject(weights, biases)
        weights_list.append(wso)
    
    logging.info(f"Created {len(weights_list)} permuted weight configurations")
    logging.info("Converting to flat tensors...")
    reference_bn_stats = bn_stats_list[0] if bn_stats_list else None

    flat_target_weights = torch.stack([wso.flatten(device) for wso in weights_list])
    flat_dim = flat_target_weights.shape[1]

    source_std = 0.001
    flat_source_weights = torch.randn(len(weights_list), flat_dim, device=device) * source_std
    
    source_dataset = TensorDataset(flat_source_weights)
    target_dataset = TensorDataset(flat_target_weights)

    sourceloader = DataLoader(source_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    targetloader = DataLoader(target_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    cfm = train_weight_space_flow(sourceloader, targetloader, flat_dim)
            
    logging.info("Generating new weights...")
    n_samples = 100
    
    random_flat = torch.randn(n_samples, flat_dim, device=device) * source_std
    new_weights_flat = cfm.map(random_flat, n_steps=60, method="rk4")

    for i in range(n_samples):
        new_wso = WeightSpaceObject.from_flat(
            new_weights_flat[i],
            weight_shapes,
            bias_shapes,
            device=device
        )

        model = ResNet20()
        idx_w = 0
        idx_b = 0
        
        model.conv1.weight.data = new_wso.weights[idx_w].clone(); idx_w += 1
        model.bn1.weight.data = new_wso.weights[idx_w].clone(); idx_w += 1
        model.bn1.bias.data = new_wso.biases[idx_b].clone(); idx_b += 1
        
        for block in range(3):
            model.layer1[block].conv1.weight.data = new_wso.weights[idx_w].clone(); idx_w += 1
            model.layer1[block].bn1.weight.data = new_wso.weights[idx_w].clone(); idx_w += 1
            model.layer1[block].bn1.bias.data = new_wso.biases[idx_b].clone(); idx_b += 1
            model.layer1[block].conv2.weight.data = new_wso.weights[idx_w].clone(); idx_w += 1
            model.layer1[block].bn2.weight.data = new_wso.weights[idx_w].clone(); idx_w += 1
            model.layer1[block].bn2.bias.data = new_wso.biases[idx_b].clone(); idx_b += 1
            
        for block in range(3):
            if block == 0:
                model.layer2[block].shortcut[0].weight.data = new_wso.weights[idx_w].clone(); idx_w += 1
                model.layer2[block].shortcut[1].weight.data = new_wso.weights[idx_w].clone(); idx_w += 1
                model.layer2[block].shortcut[1].bias.data = new_wso.biases[idx_b].clone(); idx_b += 1
            
            model.layer2[block].conv1.weight.data = new_wso.weights[idx_w].clone(); idx_w += 1
            model.layer2[block].bn1.weight.data = new_wso.weights[idx_w].clone(); idx_w += 1
            model.layer2[block].bn1.bias.data = new_wso.biases[idx_b].clone(); idx_b += 1
            model.layer2[block].conv2.weight.data = new_wso.weights[idx_w].clone(); idx_w += 1
            model.layer2[block].bn2.weight.data = new_wso.weights[idx_w].clone(); idx_w += 1
            model.layer2[block].bn2.bias.data = new_wso.biases[idx_b].clone(); idx_b += 1
        
        for block in range(3):
            if block == 0:
                model.layer3[block].shortcut[0].weight.data = new_wso.weights[idx_w].clone(); idx_w += 1
                model.layer3[block].shortcut[1].weight.data = new_wso.weights[idx_w].clone(); idx_w += 1
                model.layer3[block].shortcut[1].bias.data = new_wso.biases[idx_b].clone(); idx_b += 1
            
            model.layer3[block].conv1.weight.data = new_wso.weights[idx_w].clone(); idx_w += 1
            model.layer3[block].bn1.weight.data = new_wso.weights[idx_w].clone(); idx_w += 1
            model.layer3[block].bn1.bias.data = new_wso.biases[idx_b].clone(); idx_b += 1
            model.layer3[block].conv2.weight.data = new_wso.weights[idx_w].clone(); idx_w += 1
            model.layer3[block].bn2.weight.data = new_wso.weights[idx_w].clone(); idx_w += 1
            model.layer3[block].bn2.bias.data = new_wso.biases[idx_b].clone(); idx_b += 1
        
        model.linear.weight.data = new_wso.weights[idx_w].clone(); idx_w += 1
        model.linear.bias.data = new_wso.biases[idx_b].clone(); idx_b += 1

        model = recalibrate_bn_stats(model, test_loader, device)
        model = model.to(device)
        acc = evaluate(model, test_loader)
        logging.info(f"Rk4 Generated model {i} accuracy: {acc:.2f}%")

if __name__ == "__main__":
    main()