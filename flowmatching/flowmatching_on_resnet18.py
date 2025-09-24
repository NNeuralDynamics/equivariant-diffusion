import sys
import torch
import numpy as np
from collections import defaultdict
from typing import NamedTuple
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm
import copy
import logging
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
from torch.cuda.amp import autocast, GradScaler
from sklearn.decomposition import IncrementalPCA


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.basicConfig(stream=sys.stdout, format='%(asctime)s %(levelname)s: %(message)s', level=logging.INFO, datefmt='%I:%M:%S')

class Bunch:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

def get_cifar10_loaders(batch_size=128):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])
    
    train_ds = torchvision.datasets.CIFAR10(root="data", train=True, download=True, transform=transform_train)
    test_ds = torchvision.datasets.CIFAR10(root="data", train=False, download=True, transform=transform_test)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    return train_loader, test_loader

# ------------------- Model -------------------
def get_resnet18(num_classes=10):
    model = torchvision.models.resnet18()
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model.to(device)

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
        **conv(f"{name}.downsample.0", p_in, p_out),
        **norm(f"{name}.downsample.1", p_out),
    }

    return permutation_spec_from_axes_to_perm({
        # Initial conv
        **conv("conv1", None, "P_l0"),
        **norm("bn1", "P_l0"),

        # Layer 1 (2 blocks)
        **easyblock("layer1.0", "P_l0"),
        **easyblock("layer1.1", "P_l0"),

        # Layer 2 (2 blocks, first has shortcut)
        **shortcutblock("layer2.0", "P_l0", "P_l1"),
        **easyblock("layer2.1", "P_l1"),

        # Layer 3 (2 blocks, first has shortcut)
        **shortcutblock("layer3.0", "P_l1", "P_l2"),
        **easyblock("layer3.1", "P_l2"),

        # Layer 4 (2 blocks, first has shortcut)
        **shortcutblock("layer4.0", "P_l2", "P_l3"),
        **easyblock("layer4.1", "P_l3"),

        # Final dense layer
        **dense("fc", "P_l3", None),
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

def get_permuted_models_data(ref_point=0, model_dir="../cifar10_models", num_models=100, device=None):
    """Apply weight matching to align models with a reference model"""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    ref_model = get_resnet18()
    ref_model_path = f"{model_dir}/resnet18_seed{ref_point}.pt"
    
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
    
    permuted_models, org_models = [], []
    permuted_models.append(ref_model)
    org_models.append(ref_model)

    for i in tqdm(range(num_models), desc="Processing models"):
        if i == ref_point:
            continue
        
        model_path = f"{model_dir}/resnet18_seed{i}.pt"
        if not os.path.exists(model_path):
            logging.info(f"Skipping model {i} - file not found")
            continue
        
        try:
            # Load model B
            model_b = get_resnet18()
            model_b.load_state_dict(torch.load(model_path, map_location=device))
            model_b = model_b.to(device)
            org_models.append(model_b)
        
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
            
            permuted_models.append(reconstructed_model)
        
        except Exception as e:
            logging.error(f"Error processing model {i}: {e}")
            continue
        
        torch.cuda.empty_cache()
    
    logging.info(f"Processed {len(permuted_models)} models successfully")
    return ref_model, org_models, permuted_models

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


def evaluate(model, device='cuda' if torch.cuda.is_available() else 'cpu'):
    model.eval()
    model.to(device)
    test_loader = get_test_loader()
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


def print_stats(models):
    accuracies = []
    for i, model in enumerate(models):
        acc = evaluate(model)
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
    def __init__(self, input_dim, time_embed_dim=128):
        super().__init__()
        self.input_dim = input_dim
        self.time_embed_dim = time_embed_dim
        
        self.time_embed = nn.Sequential(
            nn.Linear(1, time_embed_dim),
            nn.GELU(),
            nn.Linear(time_embed_dim, time_embed_dim)
        )
        
        hidden_dim = 1024
        
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

# Training recommendations
def train_weight_space_flow(sourceloader, targetloader, input_dim):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    flow_model = WeightSpaceFlowModel(input_dim).to(device)
    
    t_dist = "uniform"
    cfm = FlowMatching(
        sourceloader=sourceloader,
        targetloader=targetloader,
        model=flow_model,
        mode="velocity",
        t_dist=t_dist, 
        device=device
    )
    
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
        scheduler = scheduler,
        sigma=0.001, 
        patience=100, 
        log_freq=10
    )
    
    return cfm

def recalibrate_bn_stats(model, device='cuda', print_stats=False):
    """Recalculate BatchNorm statistics for generated weights"""
    model.train()
    model.to(device)
    test_loader = get_test_loader(batch_size=128)
    
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.reset_running_stats()
            m.momentum = None
    
    with torch.no_grad():
        for inputs, _ in test_loader:
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

def assign_bn_stats_from_reference(model, ref_model):
    """Copy BN stats (mean/var) from reference model0"""
    ref_bn_layers = [m for m in ref_model.modules() if isinstance(m, nn.BatchNorm2d)]
    tgt_bn_layers = [m for m in model.modules() if isinstance(m, nn.BatchNorm2d)]
    assert len(ref_bn_layers) == len(tgt_bn_layers)
    for ref_bn, tgt_bn in zip(ref_bn_layers, tgt_bn_layers):
        tgt_bn.running_mean.data.copy_(ref_bn.running_mean.data)
        tgt_bn.running_var.data.copy_(ref_bn.running_var.data)
        tgt_bn.num_batches_tracked.data.copy_(ref_bn.num_batches_tracked.data)
    return model

def summarize_results(results):
    """Pretty-print summary of accuracies across BN strategies"""
    header = f"{'Strategy':<15} | {'Mean ± Std':<15} | {'Min':<7} | {'Max':<7}"
    line = "-" * len(header)
    print("\n" + line)
    print(header)
    print(line)

    for k, v in results.items():
        v = np.array(v)
        mean, std = v.mean(), v.std()
        vmin, vmax = v.min(), v.max()
        print(f"{k:<15} | {mean:.2f} ± {std:.2f}   | {vmin:.2f}   | {vmax:.2f}")

    print(line + "\n")

def compare_bn_strategies(generated_models, ref_model, device='cuda'):
    """Compare accuracy under 3 BN handling strategies"""
    results = {"no_calibration": [], "ref_bn": [], "recalibrated": []}

    for i, model in enumerate(generated_models):
        # (1) No calibration
        acc_no = evaluate(model.to(device), device)
        results["no_calibration"].append(acc_no)

        # (2) Reference BN assignment
        model_refbn = copy.deepcopy(model).to(device)
        model_refbn = assign_bn_stats_from_reference(model_refbn, ref_model)
        acc_ref = evaluate(model_refbn, device)
        results["ref_bn"].append(acc_ref)

        # (3) Recalibrated BN
        model_recal = copy.deepcopy(model).to(device)
        model_recal = recalibrate_bn_stats(model_recal, device)
        acc_recal = evaluate(model_recal, device)
        results["recalibrated"].append(acc_recal)

    for k, v in results.items():
        logging.info(f"{k}: mean={np.mean(v):.2f} ± {np.std(v):.2f}")

    return results

 
def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = get_resnet18()
    weight_shapes, bias_shapes = [], []
    for name, param in model.named_parameters():
        if "weight" in name:
            weight_shapes.append(tuple(param.shape))
        elif "bias" in name:
            bias_shapes.append(tuple(param.shape))
    
    print("weight_shapes =", weight_shapes)
    print("bias_shapes   =", bias_shapes)

    batch_size = 8

    logging.info("Creating permuted model dataset using rebasin...")
    ref_point = 0
    ref_model, org_models, permuted_models = get_permuted_models_data(ref_point=ref_point)
    logging.info("Orginal Models")
    print_stats(org_models)
    logging.info("Permuted Models")
    print_stats(permuted_models)

    for init_type in ["gaussian_0.01"]:
        for model_type in ["with_gitrebasin", "without_rebasin"]:
            if model_type == "with_gitrebasin":
                models_to_use = permuted_models
            else:
               models_to_use = org_models 

            logging.info("Converting models to WeightSpaceObjects...")
            weights_list = []
        
            for model in tqdm(models_to_use):
                weights = []
                biases = []
                for name, param in model.named_parameters():
                    if "weight" in name:
                        weights.append(param.data.clone())
                    elif "bias" in name:
                        biases.append(param.data.clone())
                wso = WeightSpaceObject(weights, biases)
                weights_list.append(wso)
            
            logging.info(f"Created {len(weights_list)} weight configurations")
        
            flat_target_weights = torch.stack([wso.flatten("cpu") for wso in weights_list])
            flat_dim = flat_target_weights.shape[1]

            latent_dim = 99
            ipca = IncrementalPCA(n_components=latent_dim, batch_size=10)
            flat_latent = ipca.fit_transform(flat_target_weights.cpu().numpy())

            logging.info(flat_dim)
    
            n_samples = 25
            
            if "gaussian" in init_type:
                if init_type == "gaussian_0.01":
                    source_std = 0.01
                else:
                    source_std = 0.001

                latent_tensor = torch.tensor(flat_latent, dtype=torch.float32)
                source_tensor = torch.randn_like(latent_tensor) * source_std 
                random_flat = torch.randn(n_samples, latent_dim, device=device) * source_std
            
            elif "kaimings" in init_type:
                flat_source_weights = []
                for wso in weights_list:
                    # Apply Kaiming to each weight matrix, keep biases zeros
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
    
            source_dataset = TensorDataset(source_tensor)
            target_dataset = TensorDataset(latent_tensor)
            
            sourceloader = DataLoader(source_dataset, batch_size=2, shuffle=True)
            targetloader = DataLoader(target_dataset, batch_size=2, shuffle=True)
        
            cfm = train_weight_space_flow(sourceloader, targetloader, latent_dim)
        
            logging.info("Generating new weights...")
            for gen_method in ["rk4"]:
                generated_latent = cfm.map(random_flat, n_steps=100, method=gen_method).cpu().numpy()
                new_weights_flat = ipca.inverse_transform(generated_latent)
                
                generated_models = []
        
                for i in range(n_samples):
                    new_wso = WeightSpaceObject.from_flat(
                        new_weights_flat[i],
                        weight_shapes,
                        bias_shapes,
                        device=device
                    )
                
                    model = get_resnet18()
                    
                    param_dict = {}
                    weight_idx, bias_idx = 0, 0
                    
                    for name, param in model.named_parameters():
                        if "weight" in name:
                            param_dict[name] = torch.tensor(new_wso.weights[weight_idx], device=device, dtype=param.dtype)
                            weight_idx += 1
                        elif "bias" in name:
                            param_dict[name] = torch.tensor(new_wso.biases[bias_idx], device=device, dtype=param.dtype)
                            bias_idx += 1
                    
                    with torch.no_grad():
                        for name, param in model.named_parameters():
                            if name in param_dict:
                                param.copy_(param_dict[name])
                
                    # Recalibrate BN stats
                    model = recalibrate_bn_stats(model, device)
                    model = model.to(device)
                    generated_models.append(model)
                    del model
                    
                logging.info(f"Init Type: {init_type}, Model Type: {model_type}, Generation Method: {gen_method}")
                print_stats(generated_models)

                # results = compare_bn_strategies(generated_models, ref_model, device)
                # summarize_results(results)
                
if __name__ == "__main__":
    logging.info("CIFAR-10 Resnet18 embed 1024 time 128")
    main()
