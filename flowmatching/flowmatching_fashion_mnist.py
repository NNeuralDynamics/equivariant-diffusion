import sys
import torch
import numpy as np
from collections import defaultdict
from typing import NamedTuple
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm
import copy
import logging

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.nn.functional as F
import copy
from collections import defaultdict
import os
import traceback

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.basicConfig(stream=sys.stdout, format='%(asctime)s %(levelname)s: %(message)s', level=logging.INFO, datefmt='%I:%M:%S')

class MLP(nn.Module):
    def __init__(self, init_type='xavier', seed=None):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 10)
        
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

class PermutationSpec(NamedTuple):
    perm_to_axes: dict
    axes_to_perm: dict

def permutation_spec_from_axes_to_perm(axes_to_perm: dict) -> PermutationSpec:
    perm_to_axes = defaultdict(list)
    for wk, axis_perms in axes_to_perm.items():
        for axis, perm in enumerate(axis_perms):
            if perm is not None:
                perm_to_axes[perm].append((wk, axis))
    return PermutationSpec(dict(perm_to_axes), axes_to_perm)

def mlp_permutation_spec() -> PermutationSpec:
    """Permutation spec for 3-layer MLP"""
    return permutation_spec_from_axes_to_perm({
        "fc1.weight": (None, "P_0"),
        "fc1.bias": ("P_0",),
        "fc2.weight": ("P_0", "P_1"),
        "fc2.bias": ("P_1",),
        "fc3.weight": ("P_1", None),
        "fc3.bias": (None,),
    })

def get_permuted_param(ps: PermutationSpec, perm, k: str, params, except_axis=None):
    w = params[k]
    for axis, p in enumerate(ps.axes_to_perm[k]):
        if axis == except_axis:
            continue
        if p is not None:
            w = torch.index_select(w, axis, torch.tensor(perm[p], device=w.device))
    return w

def apply_permutation(ps: PermutationSpec, perm, params):
    return {k: get_permuted_param(ps, perm, k, params) for k in params.keys()}

def update_model_weights(model, aligned_params):
    model.fc1.weight.data = aligned_params["fc1.weight"].T
    model.fc1.bias.data = aligned_params["fc1.bias"]
    model.fc2.weight.data = aligned_params["fc2.weight"].T
    model.fc2.bias.data = aligned_params["fc2.bias"]
    model.fc3.weight.data = aligned_params["fc3.weight"].T
    model.fc3.bias.data = aligned_params["fc3.bias"]

# -----------------------------
# Weight Matching
# -----------------------------
def weight_matching(ps: PermutationSpec, params_a, params_b, max_iter=100, init_perm=None, silent=True, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    params_a = {k: v.to(device) for k, v in params_a.items()}
    params_b = {k: v.to(device) for k, v in params_b.items()}
    
    perm_sizes = {p: params_a[axes[0][0]].shape[axes[0][1]] for p, axes in ps.perm_to_axes.items()}
    
    perm = {p: torch.arange(n, device=device) for p, n in perm_sizes.items()} if init_perm is None else {p: v.to(device) for p, v in init_perm.items()}
    
    perm_names = list(perm.keys())
    rng = np.random.RandomState(42)

    for _ in range(max_iter):
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
            eye_old = torch.eye(n, device=device)[perm[p]]
            eye_new = torch.eye(n, device=device)[ci]
            oldL = torch.tensordot(A, eye_old, dims=([0,1],[0,1]))
            newL = torch.tensordot(A, eye_new, dims=([0,1],[0,1]))
            progress = progress or newL > oldL + 1e-12
            perm[p] = torch.tensor(ci, device=device)
        if not progress:
            break
    return perm

# -----------------------------
# Load and Align Models
# -----------------------------
def get_permuted_models_data(ref_point=0, model_dir="../fashion_mnist_models", num_models=100, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Reference model
    ref_model = MLP()
    ref_path = f"{model_dir}/mlp_seed{ref_point}.pt"
    ref_model.load_state_dict(torch.load(ref_path, map_location=device))
    ref_model = ref_model.to(device)
    
    ps = mlp_permutation_spec()
    params_a = {k: v.T if "weight" in k else v for k, v in ref_model.state_dict().items() if k in ps.axes_to_perm}
    
    permuted_models, org_models = [ref_model], [ref_model]
    
    for i in tqdm(range(num_models), desc="Processing models"):
        if i == ref_point:
            continue
        path = f"{model_dir}/mlp_seed{i}.pt"
        if not os.path.exists(path):
            logging.info(f"Skipping model {i} - file not found")
            continue
        
        model = MLP()
        model.load_state_dict(torch.load(path, map_location=device))
        model = model.to(device)
        org_models.append(model)
        
        params_b = {k: v.T if "weight" in k else v for k, v in model.state_dict().items() if k in ps.axes_to_perm}
        perm = weight_matching(ps, params_a, params_b, device=device)
        aligned_params = apply_permutation(ps, perm, params_b)
        
        reconstructed = copy.deepcopy(model)
        update_model_weights(reconstructed, aligned_params)
        permuted_models.append(reconstructed.to(device))
    
    return ref_model, org_models, permuted_models


# Simple Bunch class for storing data
class Bunch:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

class WeightSpaceObject:
    def __init__(self, weights, biases):
        self.weights = tuple(weights)
        self.biases = tuple(biases)
        
    def flatten(self, device=None):
        flat = torch.cat([w.flatten() for w in self.weights] + 
                        [b.flatten() for b in self.biases])
        return flat.to(device) if device else flat
    
    @classmethod
    def from_flat(cls, flat, layers, device=None):
        """Create WeightSpaceObject from flattened vector and layer sizes"""
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
            weights.append(parts[i].reshape(layers[i+1], layers[i]))
            biases.append(parts[i + len(layers) - 1])
            
        return cls(weights, biases)
        
def safe_deflatten(flat, batch_size, starts, ends):
    """Safely deflatten a tensor without index errors"""
    parts = []
    actual_batch_size = flat.size(0)
    
    safe_batch_size = min(actual_batch_size, batch_size)
    
    for i in range(safe_batch_size):
        batch_parts = []
        for si, ei in zip(starts, ends):
            if si < ei: 
                batch_parts.append(flat[i][si:ei])
        parts.append(batch_parts)
    
    return parts

class FlowMatching:
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
        # Device
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.sourceloader = sourceloader
        self.targetloader = targetloader
        self.model = model.to(self.device)
        self.mode = mode
        self.t_dist = t_dist
        self.sigma = 0.001
        self.normalize_pred = normalize_pred
        self.geometric = geometric

        # Metrics tracking
        self.metrics = {"train_loss": [], "time": [], "grad_norm": [], "flow_norm": [], "true_norm": []}
        self.best_loss = float('inf')
        self.best_model_state = None

        self.input_dim = None

    # ---------------- Sampling ----------------
    def sample_from_loader(self, loader):
        """Sample a batch from dataloader"""
        try:
            if not hasattr(loader, '_iterator') or loader._iterator is None:
                loader._iterator = iter(loader)
            try:
                batch = next(loader._iterator)
            except StopIteration:
                loader._iterator = iter(loader)
                batch = next(loader._iterator)
            return batch[0].to(self.device)
        except Exception as e:
            logging.info(f"Error sampling from loader: {str(e)}")
            if hasattr(loader.dataset, '__getitem__'):
                dummy = loader.dataset[0][0]
                return torch.zeros(loader.batch_size, *dummy.shape, device=self.device)
            return torch.zeros(loader.batch_size, 1, device=self.device)

    def sample_time_and_flow(self):
        """Sample time t and flow (for velocity or target mode)"""
        x0 = self.sample_from_loader(self.sourceloader)
        x1 = self.sample_from_loader(self.targetloader)
        batch_size = min(x0.size(0), x1.size(0))
        x0, x1 = x0[:batch_size], x1[:batch_size]

        if self.t_dist == "beta":
            alpha, beta_param = 2.0, 5.0
            t = torch.distributions.Beta(alpha, beta_param).sample((batch_size,)).to(self.device)
        else:
            t = torch.rand(batch_size, device=self.device)

        t_pad = t.view(-1, *([1] * (x0.dim() - 1)))
        mu_t = (1 - t_pad) * x0 + t_pad * x1
        epsilon = torch.randn_like(x0) * self.sigma
        xt = mu_t + epsilon
        ut = x1 - x0

        return Bunch(t=t.unsqueeze(-1), x0=x0, xt=xt, x1=x1, ut=ut, eps=epsilon, batch_size=batch_size)

    def forward(self, flow):
        flow_pred = self.model(flow.xt, flow.t)
        return None, flow_pred

    def loss_fn(self, flow_pred, flow):
        if self.mode == "target":
            l_flow = torch.mean((flow_pred.squeeze() - flow.x1) ** 2)
        else:
            l_flow = torch.mean((flow_pred.squeeze() - flow.ut) ** 2)
        return None, l_flow

    def vector_field(self, xt, t):
        """Compute vector field at point xt and time t"""
        _, pred = self.forward(Bunch(xt=xt, t=t, batch_size=xt.size(0)))
        return pred if self.mode == "velocity" else pred - xt

    def train(self, n_iters=10, optimizer=None, scheduler=None, sigma=0.001, patience=1e99, log_freq=5):
        self.sigma = sigma
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
                    if scheduler: scheduler.step()

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
                    true_tensor = flow.ut if self.mode == "velocity" else flow.x1
                    grad_norm = self.get_grad_norm()
                    self.metrics["train_loss"].append(loss.item())
                    self.metrics["flow_norm"].append(flow_pred.norm(p=2, dim=1).mean().item())
                    self.metrics["time"].append(flow.t.mean().item())
                    self.metrics["true_norm"].append(true_tensor.norm(p=2, dim=1).mean().item())
                    self.metrics["grad_norm"].append(grad_norm)
                    pbar.set_description(f"Iters [loss {loss.item():.6f}, ∇ norm {grad_norm:.6f}]")
            except Exception as e:
                logging.info(f"Error during training iteration {i}: {str(e)}")
                traceback.print_exc()
                continue

    def get_grad_norm(self):
        total = 0
        for p in self.model.parameters():
            if p.grad is not None:
                total += p.grad.detach().norm(2).item() ** 2
        return total ** 0.5

    def map(self, x0, n_steps=50, return_traj=False, method="euler"):
        if self.best_model_state is not None:
            current_state = {k: v.clone() for k, v in self.model.state_dict().items()}
            self.model.load_state_dict(self.best_model_state)

        self.model.eval()
        batch_size, flat_dim = x0.size()
        traj = [x0.detach().clone()] if return_traj else None
        xt = x0.clone()
        times = torch.linspace(0, 1, n_steps, device=self.device)
        dt = times[1] - times[0]

        for i, t in enumerate(times[:-1]):
            with torch.no_grad():
                t_tensor = torch.ones(batch_size, 1, device=self.device) * t
                pred = self.model(xt, t_tensor)
                if pred.dim() > 2: pred = pred.squeeze(-1)
                vt = pred if self.mode == "velocity" else pred - xt
                if method == "euler":
                    xt = xt + vt * dt
                elif method == "rk4":
                    # RK4 steps
                    k1 = vt
                    k2 = self.model(xt + 0.5 * dt * k1, t_tensor + 0.5 * dt)
                    if k2.dim() > 2: k2 = k2.squeeze(-1)
                    k2 = k2 if self.mode == "velocity" else k2 - (xt + 0.5 * dt * k1)
                    k3 = self.model(xt + 0.5 * dt * k2, t_tensor + 0.5 * dt)
                    if k3.dim() > 2: k3 = k3.squeeze(-1)
                    k3 = k3 if self.mode == "velocity" else k3 - (xt + 0.5 * dt * k2)
                    k4 = self.model(xt + dt * k3, t_tensor + dt)
                    if k4.dim() > 2: k4 = k4.squeeze(-1)
                    k4 = k4 if self.mode == "velocity" else k4 - (xt + dt * k3)
                    xt = xt + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

                if return_traj:
                    traj.append(xt.detach().clone())

        if self.best_model_state is not None:
            self.model.load_state_dict(current_state)
        self.model.train()
        return traj if return_traj else xt

    def generate_weights(self, n_samples=10, source_noise_std=0.001, **map_kwargs):
        assert self.input_dim is not None, "Set `self.input_dim` before generating weights."
        source_samples = torch.randn(n_samples, self.input_dim, device=self.device) * source_noise_std
        return self.map(source_samples, **map_kwargs)

    def plot_metrics(self):
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


class WeightSpaceFlowModel(nn.Module):
    def __init__(self, input_dim, time_embed_dim=64):
        super().__init__()
        self.input_dim = input_dim
        self.time_embed_dim = time_embed_dim
        
        self.time_embed = nn.Sequential(
            nn.Linear(1, time_embed_dim),
            nn.GELU(),
            nn.Linear(time_embed_dim, time_embed_dim)
        )
        
        hidden_dim = min(512, input_dim // 4)
        logging.info(f"hidden_dim:{hidden_dim}")
        
        self.net = nn.Sequential(
            nn.Linear(input_dim + time_embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.LayerNorm(hidden_dim//2), 
            nn.GELU(),
            nn.Dropout(0.1),
            
            nn.Linear(hidden_dim//2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            
            nn.Linear(hidden_dim, input_dim)
        )
        
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
    
    def forward(self, x, t):
        t_embed = self.time_embed(t)
        combined = torch.cat([x, t_embed], dim=-1)
        return self.net(combined)


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

def print_stats(models, test_loader):
    accuracies = []
    for i, model in enumerate(models):
        acc = test_mlp(model, test_loader)
        accuracies.append(acc)

    accuracies = np.array(accuracies)
    mean = accuracies.mean()
    std = accuracies.std()
    min_acc = accuracies.min()
    max_acc = accuracies.max()

    logging.info("\n=== Summary ===")
    logging.info(f"Average Accuracy: {mean:.2f}% ± {std:.2f}%")
    logging.info(f"Min Accuracy: {min_acc:.2f}%")
    logging.info(f"Max Accuracy: {max_acc:.2f}%")

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main():
    layer_layout = [784, 128, 128, 10]
    batch_size = 8
    
    transform = transforms.Compose([transforms.ToTensor()])
    test_data = datasets.FashionMNIST(root="data", train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=32, shuffle=False)
  
    logging.info("Creating permuted model dataset using rebasin...")
    ref_point = 12

    ref_model, org_models, permuted_models = get_permuted_models_data(ref_point=ref_point)
    logging.info("Orginal Models")
    print_stats(org_models, test_loader)
    logging.info("Permuted Models")
    print_stats(permuted_models, test_loader)

    for init_type in ["gaussian_0.01"]:
        for model_type in ["with_gitrebasin", "without_rebasin"]:
            if model_type == "with_gitrebasin":
                models_to_use = permuted_models
            else:
               models_to_use = org_models 
    
            logging.info("Converting models to WeightSpaceObjects...")
            weights_list = []
            for model in tqdm(models_to_use):
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
            
            logging.info("Converting to flat tensors...")
            flat_target_weights = torch.stack([wso.flatten(device) for wso in weights_list])
            flat_dim = flat_target_weights.shape[1]
            
            n_samples = 25
            
            if "gaussian" in init_type:
                if init_type == "gaussian_0.01":
                    source_std = 0.01
                else:
                    source_std = 0.001
            
                flat_source_weights = torch.randn(len(weights_list), flat_dim, device=device) * source_std
                random_flat = torch.randn(n_samples, flat_dim, device=device) * source_std
            
            elif "kaimings" in init_type:
                flat_source_weights = []
                for wso in weights_list:
                    kaiming_weights = []
                    for w in wso.weights:
                        fan_in = w.shape[1]
                        std = np.sqrt(2.0 / fan_in)
                        kaiming_w = torch.randn_like(w) * std
                        kaiming_weights.append(kaiming_w)
                    
                    biases = [b.clone() for b in wso.biases]
                    
                    flat_source_weights.append(WeightSpaceObject(kaiming_weights, biases).flatten(device))
                
                flat_source_weights = torch.stack(flat_source_weights)
                
                fan_in_global = weights_list[0].weights[0].shape[1]
                kaiming_std = np.sqrt(2.0 / fan_in_global)
                random_flat = torch.randn(n_samples, flat_dim, device=device) * kaiming_std
    
            else:
                raise ValueError(f"Unknown init_type: {init_type}")
                
            source_dataset = TensorDataset(flat_source_weights)
            target_dataset = TensorDataset(flat_target_weights)
            
            sourceloader = DataLoader(source_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
            targetloader = DataLoader(target_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        
            flow_model = WeightSpaceFlowModel(flat_dim).to(device)
            flow_model.train()
            
            t_dist = "uniform"
            cfm = FlowMatching(
                sourceloader=sourceloader,
                targetloader=targetloader,
                model=flow_model,
                mode="velocity",
                t_dist=t_dist, 
                device=device
            )
            
            n_params_base = sum(p.numel() for p in MLP().parameters())
            n_params_flow = count_parameters(flow_model)
            logging.info(f"MLP params:{n_params_base}")
            logging.info(f"Flow model params:{n_params_flow}")
        
            optimizer = torch.optim.AdamW(
                flow_model.parameters(), 
                lr=5e-4, 
                weight_decay=1e-5,
                betas=(0.9, 0.95)
            )
            
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=30000, eta_min=1e-6
            )
            
            cfm.train(
                n_iters=30000,
                optimizer=optimizer,
                scheduler=scheduler,
                sigma=0.001,
                patience=100,
                log_freq=10
            )
            
            logging.info("Generating new MLP weights...")
                        
            for gen_method in ["rk4"]:
                new_weights_flat = cfm.map(random_flat, n_steps=100, method=gen_method)
                generated_models = []        
    
                for i in range(n_samples):
                    new_wso = WeightSpaceObject.from_flat(
                        new_weights_flat[i], 
                        layers=np.array(layer_layout), 
                        device=device
                    )
            
                    expected_weight_shapes = [(128, 784), (128, 128), (10, 128)]
                    expected_bias_shapes = [(128,), (128,), (10,)]
            
                    assert len(new_wso.weights) == 3, f"Expected 3 weight matrices, got {len(new_wso.weights)}"
                    assert len(new_wso.biases) == 3, f"Expected 3 bias vectors, got {len(new_wso.biases)}"
                    
                    for j, (w, expected_shape) in enumerate(zip(new_wso.weights, expected_weight_shapes)):
                        assert w.shape == expected_shape, f"Weight {j} has shape {w.shape}, expected {expected_shape}"
                    
                    for j, (b, expected_shape) in enumerate(zip(new_wso.biases, expected_bias_shapes)):
                        assert b.shape == expected_shape, f"Bias {j} has shape {b.shape}, expected {expected_shape}"
            
                    model = MLP()
                    model.fc1.weight.data = new_wso.weights[0].clone()
                    model.fc1.bias.data = new_wso.biases[0].clone()
                    model.fc2.weight.data = new_wso.weights[1].clone()
                    model.fc2.bias.data = new_wso.biases[1].clone()
                    model.fc3.weight.data = new_wso.weights[2].clone()
                    model.fc3.bias.data = new_wso.biases[2].clone()
    
                    generated_models.append(model)
                    
                logging.info(f"Init Type: {init_type}, Model Type: {model_type}, Generation Method: {gen_method}")
                print_stats(generated_models, test_loader)
        
if __name__ == "__main__":
    logging.info("MLP - Fashion MNIST embed 512")
    main()
