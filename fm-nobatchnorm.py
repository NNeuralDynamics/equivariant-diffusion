import sys

sys.path.insert(1, "..")

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms, datasets
import torchvision
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from collections import defaultdict, namedtuple
from typing import NamedTuple
from scipy.optimize import linear_sum_assignment
import logging
import copy
import traceback
from utils.data import sample_gaussian_wsos
from torch.optim.lr_scheduler import CosineAnnealingLR

# Set up device and logging
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.basicConfig(
    stream=sys.stdout, 
    format='%(asctime)s %(levelname)s: %(message)s', 
    level=logging.INFO, 
    datefmt='%I:%M:%S'
)

# Simple Bunch class for data containers
class Bunch:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

# RESNET MODEL DEFINITION #

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride, padding=1, bias=False)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=1, padding=1, bias=False)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = self.conv2(out)
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super().__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
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
        out = F.relu(self.conv1(x))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def ResNet20():
    return ResNet(BasicBlock, [3,3,3])

# WEIGHT MATCHING #


# PermutationSpec class for defining permutable dimensions
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
    
    dense = lambda name, p_in, p_out: {
        f"{name}.weight": (p_out, p_in),
        f"{name}.bias": (p_out,)
    }

    easyblock = lambda name, p: {
        **conv(f"{name}.conv1", p, f"P_{name}_inner"),
        **conv(f"{name}.conv2", f"P_{name}_inner", p),
    }

    shortcutblock = lambda name, p_in, p_out: {
        **conv(f"{name}.conv1", p_in, f"P_{name}_inner"),
        **conv(f"{name}.conv2", f"P_{name}_inner", p_out),
        **conv(f"{name}.shortcut.0", p_in, p_out),
    }

    return permutation_spec_from_axes_to_perm({
        **conv("conv1", None, "P_bg0"),

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

    # Get permutation sizes
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

#  WEIGHT SPACE OBJECT  #

class WeightSpaceObject:
    def __init__(self, weights, biases):
        self.weights = weights if isinstance(weights, tuple) else tuple(weights)
        self.biases = biases if isinstance(biases, tuple) else tuple(biases)
        
    def flatten(self, device=None):
        """Flatten weights and biases into a single vector"""
        flat = torch.cat(
            [w.reshape(-1) for w in self.weights] +
            [b.reshape(-1) for b in self.biases]
        )
        return flat.to(device) if device else flat

    @classmethod
    def from_flat(cls, flat, layer_shapes, bias_shapes, device=None):
        flat = flat.to(device) if device else flat
        sizes = [np.prod(s) for s in layer_shapes + bias_shapes]
    
        if flat.numel() != sum(sizes):
            raise ValueError(f"Expected flat vector of length {sum(sizes)}, got {flat.numel()}")
    
        parts = []
        start = 0
        for size in sizes:
            parts.append(flat[start:start+size])
            start += size
    
        weights = [parts[i].reshape(layer_shapes[i]) for i in range(len(layer_shapes))]
        biases = [parts[len(layer_shapes) + i].reshape(bias_shapes[i]) for i in range(len(bias_shapes))] if bias_shapes else []

        return cls(weights, biases).to(device)


    def to(self, device):
        weights = tuple(w.to(device) for w in self.weights)
        biases = tuple(b.to(device) for b in self.biases)
        return WeightSpaceObject(weights, biases)

    def map(self, fn):
            new_weights = tuple(fn(w) for w in self.weights)
            new_biases = tuple(fn(b) for b in self.biases)
            return WeightSpaceObject(new_weights, new_biases)

def create_zero_wso(template_wso):
    zero_weights = [torch.zeros_like(w) for w in template_wso.weights]
    zero_biases = [torch.zeros_like(b) for b in template_wso.biases]
    return WeightSpaceObject(zero_weights, zero_biases)


# EVALUATION UTILS #

def get_test_loader(batch_size=128):
    """Create a test data loader for CIFAR-10"""
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    test_set = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test
    )

    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2)
    return test_loader

def evaluate(model, test_loader, device=None):
    """Evaluate model accuracy on test data"""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
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

# FLOW MATCHING (SIMPLIFIED) #

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
        elif self.fm_type == "ot":
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
        dt = times[1] - times[0]  # Time step size

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
        scheduler = CosineAnnealingLR(optimizer, T_max=n_iters, eta_min=1e-4)
        
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
                    scheduler.step()
                
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

# FLOW MODEL ARCHITECTURE #

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

# MAIN FUNCTIONS #

def get_permuted_models_data(ref_point=20, model_dir="dummy_imagenet_resnet_models", num_models=501, device=None):
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
            
            # Evaluate accuracy before adding to list
            # test_loader = get_test_loader()
            # accuracy = evaluate(reconstructed_model, test_loader)
            # logging.info(f"Model {i} accuracy after matching: {accuracy:.2f}%")
            
            permuted_models.append(reconstructed_model)
        
        except Exception as e:
            logging.error(f"Error processing model {i}: {e}")
            continues
        
        torch.cuda.empty_cache()
    
    logging.info(f"Processed {len(permuted_models)} models successfully")
    return ref_model, permuted_models

def zero_like_wso(wso):
    zero_weights = tuple(torch.zeros_like(w) for w in wso.weights)
    zero_biases = tuple(torch.zeros_like(b) for b in wso.biases)
    return WeightSpaceObject(zero_weights, zero_biases)

def get_random_pytorch_initialized_wsos(n, device):
    wsos = []
    for i in range(n):
        torch.manual_seed(i)
        model = ResNet20().to(device)
        weights = (
            model.conv1.weight.data.clone(),
            model.layer1[0].conv1.weight.data.clone(),
            model.layer1[0].conv2.weight.data.clone(),
            model.layer1[1].conv1.weight.data.clone(),
            model.layer1[1].conv2.weight.data.clone(),
            model.layer1[2].conv1.weight.data.clone(),
            model.layer1[2].conv2.weight.data.clone(),
            model.layer2[0].shortcut[0].weight.data.clone(),
            model.layer2[0].conv1.weight.data.clone(),
            model.layer2[0].conv2.weight.data.clone(),
            model.layer2[1].conv1.weight.data.clone(),
            model.layer2[1].conv2.weight.data.clone(),
            model.layer2[2].conv1.weight.data.clone(),
            model.layer2[2].conv2.weight.data.clone(),
            model.layer3[0].shortcut[0].weight.data.clone(),
            model.layer3[0].conv1.weight.data.clone(),
            model.layer3[0].conv2.weight.data.clone(),
            model.layer3[1].conv1.weight.data.clone(),
            model.layer3[1].conv2.weight.data.clone(),
            model.layer3[2].conv1.weight.data.clone(),
            model.layer3[2].conv2.weight.data.clone(),
            model.linear.weight.data.clone()
        )
        biases = (model.linear.bias.data.clone(),)
        wso = WeightSpaceObject(weights, biases)
        wsos.append(wso)
    return wsos



def train_and_generate_resnet_weights(
    num_models=501,
    batch_size=16,
    n_samples=10,
    device=None
):


    # All weight shapes
    layer_shapes = [
        (16, 3, 3, 3),  # conv1
    
        # layer1 (3 BasicBlocks)
        (16, 16, 3, 3), (16, 16, 3, 3),  # block 0
        (16, 16, 3, 3), (16, 16, 3, 3),  # block 1
        (16, 16, 3, 3), (16, 16, 3, 3),  # block 2
    
        # layer2 (3 BasicBlocks)
        (32, 16, 1, 1),                  # shortcut for layer2[0]
        (32, 16, 3, 3), (32, 32, 3, 3),  # block 0
        (32, 32, 3, 3), (32, 32, 3, 3),  # block 1
        (32, 32, 3, 3), (32, 32, 3, 3),  # block 2
    
        # layer3 (3 BasicBlocks)
        (64, 32, 1, 1),                  # shortcut for layer3[0]
        (64, 32, 3, 3), (64, 64, 3, 3),  # block 0
        (64, 64, 3, 3), (64, 64, 3, 3),  # block 1
        (64, 64, 3, 3), (64, 64, 3, 3),  # block 2
    
        (10, 64)  # final linear
    ]

    bias_shapes = [
        (10,)
    ]
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    logging.info("Getting permuted models...")
    ref_model, permuted_models = get_permuted_models_data(num_models=num_models, device=device)


    logging.info("Converting models to WeightSpaceObjects...")
    weights_list = []
    for model in tqdm(permuted_models):
        weights = (
            model.conv1.weight.data.clone(),
            model.layer1[0].conv1.weight.data.clone(),
            model.layer1[0].conv2.weight.data.clone(),
            model.layer1[1].conv1.weight.data.clone(),
            model.layer1[1].conv2.weight.data.clone(),
            model.layer1[2].conv1.weight.data.clone(),
            model.layer1[2].conv2.weight.data.clone(),
            model.layer2[0].shortcut[0].weight.data.clone(),
            model.layer2[0].conv1.weight.data.clone(),
            model.layer2[0].conv2.weight.data.clone(),
            model.layer2[1].conv1.weight.data.clone(),
            model.layer2[1].conv2.weight.data.clone(),
            model.layer2[2].conv1.weight.data.clone(),
            model.layer2[2].conv2.weight.data.clone(),
            model.layer3[0].shortcut[0].weight.data.clone(),
            model.layer3[0].conv1.weight.data.clone(),
            model.layer3[0].conv2.weight.data.clone(),
            model.layer3[1].conv1.weight.data.clone(),
            model.layer3[1].conv2.weight.data.clone(),
            model.layer3[2].conv1.weight.data.clone(),
            model.layer3[2].conv2.weight.data.clone(),
            model.linear.weight.data.clone()
        )
        
        biases = (
            model.linear.bias.data.clone(),
        )
        
        wso = WeightSpaceObject(weights, biases)
        weights_list.append(wso)
    
   
    flat_target_weights = torch.stack([wso.flatten(device) for wso in weights_list])
    flat_dim = flat_target_weights.shape[1]

    logging.info(f"Created {len(weights_list)} target weight configurations")

    # for experimenting by generating source wsos from guassian noise
    source_std = 0.01 #for noise we had 0.001

    # Uncomment this part to get source from random noise
    # flat_source_weights = torch.randn(len(weights_list), flat_dim, device=device) * source_std
    mean_zero_wso = zero_like_wso(weights_list[0])
    source_wsos = sample_gaussian_wsos(mean=mean_zero_wso, std=source_std, n=len(weights_list))
    flat_source_weights = torch.stack([wso.flatten(device) for wso in source_wsos])

    # Uncomment for experimenting by generating source wsos from kaimings Initialization
    # source_wsos = get_random_pytorch_initialized_wsos(n=len(weights_list), device = device)
    # flat_source_weights = torch.stack([wso.flatten(device) for wso in source_wsos])
    
    # Create DataLoaders
    source_dataset = TensorDataset(flat_source_weights)
    target_dataset = TensorDataset(flat_target_weights)

    sourceloader = DataLoader(source_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    targetloader = DataLoader(target_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    # Flow model
    flow_model = TimeConditionedMLP(
        input_dim=flat_dim,
        hidden_dims=[4096, 2048, 512, 128],
        output_dim=flat_dim,
    ).to(device)

    logging.info(f"ResNet20 params: {sum(p.numel() for p in ResNet20().parameters()):,}")
    logging.info(f"Flow model params: {sum(p.numel() for p in flow_model.parameters()):,}")

    optimizer = torch.optim.AdamW(flow_model.parameters(), lr=0.0001, weight_decay = 1e-6)

    # Flow Matching
    cfm =  SimpleCFM(
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
    
    logging.info("Training flow model on Random Noise...")
    cfm.train(n_iters=50000, optimizer=optimizer, log_freq=100)

    # Sampling
    logging.info("Generating new ResNet weights...")
    source_wsos = get_random_pytorch_initialized_wsos(n_samples, device=device)
    random_flat = torch.stack([
        wso.flatten(device) + torch.randn_like(wso.flatten()) * source_std
        for wso in source_wsos
    ])
    
    random_flat = torch.randn(n_samples, flat_dim, device=device) * source_std
    new_weights_flat = cfm.map(
        random_flat, 
        n_steps=100,
        noise_scale=0.0005
    )

    test_loader = get_test_loader(batch_size=128)

    for i in range(n_samples):
        new_wso = WeightSpaceObject.from_flat(
            new_weights_flat[i], 
            layer_shapes=layer_shapes, 
            bias_shapes = bias_shapes,
            device=device
        )

        model = ResNet20()

        idx = 0
        # conv1
        model.conv1.weight.data = new_wso.weights[idx].clone(); idx += 1
        
        # layer1 (3 BasicBlocks)
        for block in range(3):
            model.layer1[block].conv1.weight.data = new_wso.weights[idx].clone(); idx += 1
            model.layer1[block].conv2.weight.data = new_wso.weights[idx].clone(); idx += 1
        
        # layer2 (3 BasicBlocks)
        for block in range(3):
            if block == 0:
                # Assign shortcut (downsample) conv for first block of layer2
                model.layer2[block].shortcut[0].weight.data = new_wso.weights[idx].clone(); idx += 1
            model.layer2[block].conv1.weight.data = new_wso.weights[idx].clone(); idx += 1
            model.layer2[block].conv2.weight.data = new_wso.weights[idx].clone(); idx += 1
        
        # layer3 (3 BasicBlocks)
        for block in range(3):
            if block == 0:
                # Assign shortcut (downsample) conv for first block of layer3
                model.layer3[block].shortcut[0].weight.data = new_wso.weights[idx].clone(); idx += 1
            model.layer3[block].conv1.weight.data = new_wso.weights[idx].clone(); idx += 1
            model.layer3[block].conv2.weight.data = new_wso.weights[idx].clone(); idx += 1
        
        # Final linear layer
        model.linear.weight.data = new_wso.weights[idx].clone(); idx += 1
        model.linear.bias.data = new_wso.biases[0].clone()

            
        model = model.to(device)

        try:
            acc = evaluate(model, test_loader, device)
            logging.info(f"Generated model {i} accuracy: {acc:.2f}%")
        except Exception as e:
            logging.warning(f"Failed to evaluate model {i}: {e}")

    logging.info("Generation complete!")


def main():
    train_and_generate_resnet_weights(
        num_models=501,
        n_samples=10
    )

if __name__ == "__main__":
    main()