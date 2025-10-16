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

def get_permuted_param(ps: PermutationSpec, perm, k: str, params, except_axis=None):
    """Get parameter k from params, with permutations applied."""
    w = params[k]
    for axis, p in enumerate(ps.axes_to_perm[k]):
        if axis == except_axis:
            continue

        if p is not None:
            w = torch.index_select(w, axis, torch.tensor(perm[p], device=w.device))

    return w

def iris_mlp_permutation_spec_mlp() -> PermutationSpec:
    return permutation_spec_from_axes_to_perm({
        "fc1.weight": (None, "P_0"),       # Input (None) to hidden (P_0)
        "fc1.bias": ("P_0",),              # Bias for hidden (P_0)
        "fc2.weight": ("P_0", None),       # hidden (P_0) to output (None - no permutation)
        "fc2.bias": (None,),               # Bias for output (None - no permutation)
    })

def mnist_mlp_permutation_spec_mlp() -> PermutationSpec:
    """Define permutation spec for MLP architecture"""
    return permutation_spec_from_axes_to_perm({
        "fc1.weight": (None, "P_0"),       # Input (None) to fc1 output (P_0)
        "fc1.bias": ("P_0",),              # Bias for fc1 output (P_0)
        "fc2.weight": ("P_0", "P_1"),      # fc1 output (P_0) to fc2 output (P_1)
        "fc2.bias": ("P_1",),              # Bias for fc2 output (P_1)
        "fc3.weight": ("P_1", None),       # fc2 output (P_1) to fc3 output (None)
        "fc3.bias": (None,),               # Bias for fc3 output (None)
    }

def resnet20_without_bn_permutation_spec() -> PermutationSpec:
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
    
def resnet20_permutation_spec() -> PermutationSpec:
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

def resnet18_permutation_spec() -> PermutationSpec:
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

    # BasicBlock for ResNet18 - no expansion factor
    def basicblock(name, p_in, p_out, has_shortcut=False):
        spec = {
            **conv(f"{name}.conv1", p_in, f"P_{name}_inner"),
            **norm(f"{name}.bn1", f"P_{name}_inner"),
            **conv(f"{name}.conv2", f"P_{name}_inner", p_out),
            **norm(f"{name}.bn2", p_out),
        }
        
        # Add shortcut/downsample layers if present
        if has_shortcut:
            spec.update({
                **conv(f"{name}.downsample.0", p_in, p_out),
                **norm(f"{name}.downsample.1", p_out),
            })
        
        return spec

    return permutation_spec_from_axes_to_perm({
        # Initial conv layer
        **conv("conv1", None, "P_64"),
        **norm("bn1", "P_64"),
        
        # Layer 1: 64 -> 64 channels (no downsampling)
        **basicblock("layer1.0", "P_64", "P_64"),
        **basicblock("layer1.1", "P_64", "P_64"),
        
        # Layer 2: 64 -> 128 channels (downsampling)
        **basicblock("layer2.0", "P_64", "P_128", has_shortcut=True),
        **basicblock("layer2.1", "P_128", "P_128"),
        
        # Layer 3: 128 -> 256 channels (downsampling)
        **basicblock("layer3.0", "P_128", "P_256", has_shortcut=True),
        **basicblock("layer3.1", "P_256", "P_256"),
        
        # Layer 4: 256 -> 512 channels (downsampling)
        **basicblock("layer4.0", "P_256", "P_512", has_shortcut=True),
        **basicblock("layer4.1", "P_512", "P_512"),
        
        # Final classifier
        **dense("fc", "P_512", None),
    })

def apply_permutation(ps: PermutationSpec, perm, params):
    """Apply permutation to params"""
    return {k: get_permuted_param(ps, perm, k, params) for k in params.keys()}


def update_model_weights(model, aligned_params):
    """Update model weights with aligned parameters"""
    # Convert numpy arrays to torch tensors if needed
    model.fc1.weight.data = aligned_params["fc1.weight"].T
    model.fc1.bias.data = aligned_params["fc1.bias"]
    model.fc2.weight.data = aligned_params["fc2.weight"].T
    model.fc2.bias.data = aligned_params["fc2.bias"]

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