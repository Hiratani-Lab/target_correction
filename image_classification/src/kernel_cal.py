from torch.func import jacrev, vjp, jvp, vmap, functional_call
import math
import torch.nn as nn
import torch
import numpy as np


def kernel_element_stats(K, name="K"):
    K = np.asarray(K)
    finite = np.isfinite(K)
    print(f"\n[{name}] shape={K.shape}, dtype={K.dtype}")
    print("  finite:", finite.all(), "  #nan:", np.isnan(K).sum(), "  #inf:", np.isinf(K).sum())

    absK = np.abs(K[finite])
    print("  abs min/median/mean/max:",
          np.min(absK), np.median(absK), np.mean(absK), np.max(absK))

    diag = np.diag(K)
    off = K[~np.eye(K.shape[0], dtype=bool)]
    print("  diag  min/median/mean/max:",
          np.min(diag), np.median(diag), np.mean(diag), np.max(diag))
    print("  off   min/median/mean/max:",
          np.min(off), np.median(off), np.mean(off), np.max(off))

    sym_err = np.max(np.abs(K - K.T))
    print("  symmetry max|K-K^T|:", sym_err)


def kernel_eigen_stats(K, name="K", topk=10):
    K = np.asarray(K)
    Ks = 0.5 * (K + K.T)

    evals = np.linalg.eigvalsh(Ks)
    evals_sorted = np.sort(evals)

    print(f"\n[{name}] eigenvalues (of symmetrized K):")
    print("  min:", evals_sorted[0])
    print("  1%:", np.percentile(evals_sorted, 1))
    print("  median:", np.median(evals_sorted))
    print("  99%:", np.percentile(evals_sorted, 99))
    print("  max:", evals_sorted[-1])

    neg_mass = evals_sorted[evals_sorted < 0]
    print("  #negative:", neg_mass.size,
          "  most negative:", (neg_mass[0] if neg_mass.size else 0.0))

    pos = evals_sorted[evals_sorted > 1e-12]
    if pos.size >= 2:
        cond = pos[-1] / pos[0]
        print("  cond approx (max/min positive):", cond)
    else:
        print("  cond approx: not enough positive eigenvalues")

    print(f"  top-{topk} largest evals:", evals_sorted[-topk:])


def calc_kernels_jax(kernel_fn, Xtrain, Xtest, hy_params):
    Ntrain, Ntest, Nb = hy_params['Ntrain'], hy_params['Ntest'], hy_params['Nb']

    Kdata = np.zeros((Ntrain, Ntrain))
    Kpred = np.zeros((Ntrain, Ntest))
    print("block number:", int(Ntrain/Nb))
    for i in range(Ntrain // Nb):
        for j in range(Ntrain // Nb):
            print("kernel calculation block i*Nb+j:", i*Nb+j)
            Kdata[i * Nb:(i + 1) * Nb, j * Nb:(j + 1) * Nb] = kernel_fn(Xtrain[i * Nb:(i + 1) * Nb],
                                                                        Xtrain[j * Nb:(j + 1) * Nb], 'ntk')
            if i == 0 and j == 0:
                print("The shape of the kernel_fn:",
                      kernel_fn(Xtrain[i * Nb:(i + 1) * Nb], Xtrain[j * Nb:(j + 1) * Nb], 'ntk').shape)
        for j in range(Ntest // Nb):
            Kpred[i * Nb:(i + 1) * Nb, j * Nb:(j + 1) * Nb] = kernel_fn(Xtrain[i * Nb:(i + 1) * Nb],
                                                                        Xtest[j * Nb:(j + 1) * Nb], 'ntk')
    kernel_element_stats(Kdata, "Kdata")
    kernel_eigen_stats(Kdata, "Kdata")
    return Kdata, Kpred


def ntk_feature_single(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    model.zero_grad(set_to_none=True)
    if x.dim() == 3:
        x = x.unsqueeze(0)  # (1,1,28,28)
    logits = model(x).squeeze(0)  # (10,)

    params = [p for p in model.parameters() if p.requires_grad]
    grads_per_class = []

    for c in range(10):
        g = torch.autograd.grad(
            outputs=logits[c],
            inputs=params,
            retain_graph=True,
            create_graph=False,
            allow_unused=False,
        )
        g_flat = torch.cat([gi.reshape(-1) for gi in g], dim=0)
        grads_per_class.append(g_flat)

    feat = torch.cat(grads_per_class, dim=0)
    return feat.detach()


def compute_ntk_features(model: nn.Module, X: torch.Tensor, device="cpu", print_every=50) -> torch.Tensor:
    model.to(device)
    model.eval()
    feats = []
    for i in range(X.shape[0]):
        feat = ntk_feature_single(model, X[i].to(device))
        feats.append(feat.cpu())
        if print_every and (i + 1) % print_every == 0:
            print(f"  NTK feats: {i+1}/{X.shape[0]}")
    return torch.stack(feats, dim=0)  # (N,D)


def kernel_from_features(G1: torch.Tensor, G2: torch.Tensor, C: int, Class_avg: bool) -> np.ndarray:
    if Class_avg:
        return ((G1 @ G2.T) / C).cpu().numpy().astype(np.float64)
    else:
        return (G1 @ G2.T).cpu().numpy().astype(np.float64)


def params_dict(model):
    return {k: v for k, v in model.named_parameters()}

def buffers_dict(model):
    return {k: v for k, v in model.named_buffers()}


def make_per_sample_phi_fn(model):
    # 1. Filter parameters: ONLY grab the dense layer parameters
    p0 = {k: v for k, v in model.named_parameters() if v.requires_grad}
    #p0 = params_dict(model)
    # Grab buffers (needed for BatchNorm running stats in the frozen CNN)
    b0 = dict(model.named_buffers())
    #b0 = buffers_dict(model)

    def f(p, x):
        # logits for a single sample -> shape (C,)
        # functional_call uses `p` for the dense layer and the original model weights for the rest
        return functional_call(model, (p, b0), (x,)).squeeze(0)

    # 2. jacobian wrt params (which is p0, the first argument)
    jac_f = jacrev(f)

    def phi_single(x):
        J = jac_f(p0, x)  # dict: each value shape (C, *param_shape)
        chunks = []
        for v in J.values():
            chunks.append(v.reshape(v.shape[0], -1))  # (C, param_numel)
        Jcp = torch.cat(chunks, dim=1)  # (C, P_dense)
        return Jcp.reshape(-1)  # (C*P_dense,)

    return phi_single

def build_w_scale_map_smallcnn_staxlike(model, conv_bias_scale=1.0, dense_bias_scale=1.0):
    w_scale_map = {}

    # ---- conv1 ----
    conv1 = model.net[0]  # nn.Conv2d(1 -> 32, k=3x3)
    n1 = conv1.in_channels * conv1.kernel_size[0] * conv1.kernel_size[1]
    w_scale_map["net.0.weight"] = 1.0 / math.sqrt(n1)
    if conv1.bias is not None:
        w_scale_map["net.0.bias"] = conv_bias_scale

    # ---- conv2 ----
    conv2 = model.net[3]  # nn.Conv2d(32 -> 64, k=3x3)
    n2 = conv2.in_channels * conv2.kernel_size[0] * conv2.kernel_size[1]
    w_scale_map["net.3.weight"] = 1.0 / math.sqrt(n2)
    if conv2.bias is not None:
        w_scale_map["net.3.bias"] = conv_bias_scale

    # ---- linear ----
    fc = model.net[7]     # nn.Linear(64*7*7 -> num_classes)
    n3 = fc.in_features
    w_scale_map["net.7.weight"] = 1.0 / math.sqrt(n3)
    if fc.bias is not None:
        # If your stax Dense(..., b_std=0.) then bias is effectively not used.
        # Setting 0.0 here makes the Jacobian block contribute nothing.
        w_scale_map["net.7.bias"] = dense_bias_scale

    return w_scale_map

def make_per_sample_phi_fn_scaled(model, w_scale_map=None):
    p0 = params_dict(model)
    b0 = buffers_dict(model)

    def f(p, x):
        return functional_call(model, (p, b0), (x,)).squeeze(0)  # (C,)

    jac_f = jacrev(f)

    # fixed parameter order consistent with p0.values()
    param_names = list(p0.keys())

    def phi_single(x):
        J = jac_f(p0, x)  # dict: name -> (C, *param_shape)
        chunks = []
        for name in param_names:
            v = J[name]  # (C, *shape)
            v = v.reshape(v.shape[0], -1)  # (C, param_numel)

            if w_scale_map is not None and name in w_scale_map:
                v = v * float(w_scale_map[name])  # apply ntk scaling to Jacobian

            chunks.append(v)

        Jcp = torch.cat(chunks, dim=1)   # (C, P)
        return Jcp.reshape(-1)           # (C*P,)

    return phi_single

def make_per_sample_Jcp_fn_classwise(model, w_scale_map=None):
    model.eval()
    b0 = dict(model.named_buffers())  # empty for your net, but fine
    param_names = [k for k, _ in model.named_parameters()]

    def f(p, x):
        # x -> (1,1,28,28)
        if x.ndim == 3:              # (1,28,28)
            x_4d = x.unsqueeze(0)    # -> (1,1,28,28)
        elif x.ndim == 4:            # (1,1,28,28)
            x_4d = x
        else:
            raise ValueError(f"Unexpected x.shape={x.shape}")

        y = functional_call(model, (p, b0), (x_4d,))  # (1, C)
        return y.squeeze(0)                           # (C,)

    jac_f = jacrev(f)  # dict(name -> (C, *param_shape))

    def Jcp_single(x):
        # refresh params every call (important if model changes over time)
        p = dict(model.named_parameters())

        J = jac_f(p, x)  # dict: name -> (C, *param_shape)
        chunks = []
        for name in param_names:
            v = J[name].reshape(J[name].shape[0], -1)  # (C, numel)
            if w_scale_map is not None and name in w_scale_map:
                v = v * float(w_scale_map[name])
            chunks.append(v)
        return torch.cat(chunks, dim=1)  # (C, P)

    return Jcp_single

@torch.no_grad()
def empirical_ntk_blocks_classwise(
    model,
    X1: torch.Tensor,
    X2: torch.Tensor,
    hy_params=None,
    w_scale_map=None,
    store_cpu: bool = True,
    dtype: torch.dtype = torch.float64,
):
    block = hy_params['Nb']; device = hy_params['device']

    if device is None:
        device = next(model.parameters()).device

    model = model.to(device).eval()
    X1 = X1.to(device)
    X2 = X2.to(device)

    Jcp_single = make_per_sample_Jcp_fn_classwise(model, w_scale_map=w_scale_map)
    Jcp_batch = vmap(Jcp_single)

    N1, N2 = X1.shape[0], X2.shape[0]

    out_dev = "cpu" if store_cpu else device
    C = model(X1[:1]).shape[-1]

    Kc = torch.empty((C, N1, N2), device=out_dev, dtype=dtype)

    for i0 in range(0, N1, block):
        i1 = min(i0 + block, N1)
        Phi1 = Jcp_batch(X1[i0:i1]).to(dtype)  # (bi, C, P)

        for j0 in range(0, N2, block):
            j1 = min(j0 + block, N2)
            Phi2 = Jcp_batch(X2[j0:j1]).to(dtype)  # (bj, C, P)

            K_block = torch.einsum("icp,jcp->cij", Phi1, Phi2)

            if store_cpu:
                Kc[:, i0:i1, j0:j1] = K_block.cpu()
            else:
                Kc[:, i0:i1, j0:j1] = K_block

    return Kc

def _flatten_grads(grads, params):
    flats = []
    for g, p in zip(grads, params):
        if g is None:
            flats.append(torch.zeros_like(p).reshape(-1))
        else:
            flats.append(g.reshape(-1))
    return torch.cat(flats, dim=0)

def ntk_feature_single_classwise(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    model.zero_grad(set_to_none=True)

    if x.dim() == 3:  # (1,28,28) -> add batch dim
        x = x.unsqueeze(0)  # (1,1,28,28) if already has channel

    logits = model(x).squeeze(0)  # (C,)
    C = logits.shape[0]

    params = [p for p in model.parameters() if p.requires_grad]

    grads_per_class = []
    for c in range(C):
        grads = torch.autograd.grad(
            outputs=logits[c],
            inputs=params,
            retain_graph=(c < C - 1),   # only retain until last class
            create_graph=False,
            allow_unused=True,          # safer; some params might be unused
        )
        g_flat = _flatten_grads(grads, params)  # (P,)
        grads_per_class.append(g_flat)

    return torch.stack(grads_per_class, dim=0).detach()  # (C, P)

try:
    from functorch import vmap
except ImportError:
    vmap = torch.vmap  # PyTorch 2.0+

def empirical_ntk_blocks_classwise2(
    model,
    X1: torch.Tensor,
    X2: torch.Tensor,
    hy_params=None,
    store_cpu: bool = True,
):
    param = next(model.parameters())
    dtype = param.dtype

    block = hy_params['Nb']
    device = hy_params.get('device', None)
    if device is None:
        device = next(model.parameters()).device

    model = model.to(device).eval()
    X1 = X1.to(device)
    X2 = X2.to(device)

    def Jcp_batch_loop(model, X, dtype=torch.float32):
        # returns (B, C, P)
        outs = []
        with torch.enable_grad():
            for x in X:
                outs.append(ntk_feature_single_classwise(model, x).to(dtype))  # (C,P)
        return torch.stack(outs, dim=0)  # (B,C,P)

    N1, N2 = X1.shape[0], X2.shape[0]
    # Determine C by running once
    with torch.no_grad():
        C = model(X1[:1]).shape[-1]

    out_dev = "cpu" if store_cpu else device
    Kc = torch.empty((C, N1, N2), device=out_dev, dtype=dtype)

    for i0 in range(0, N1, block):
        i1 = min(i0 + block, N1)
        Phi1 = Jcp_batch_loop(model, X1[i0:i1], dtype=dtype) # (bi, C, P)

        for j0 in range(0, N2, block):
            j1 = min(j0 + block, N2)
            Phi2 = Jcp_batch_loop(model, X2[j0:j1], dtype=dtype)  # (bj, C, P)

            # class-wise inner products -> (C, bi, bj)
            K_block = torch.einsum("icp,jcp->cij", Phi1, Phi2)

            if store_cpu:
                Kc[:, i0:i1, j0:j1] = K_block.cpu()
            else:
                Kc[:, i0:i1, j0:j1] = K_block

    return Kc

def sample_average_phi(
    hy_params,
    model,
    X,
    block: int = 32,
    average_over_outputs: bool = False,
    return_cp: bool = False,
    store_cpu: bool = True,
    dtype: torch.dtype = torch.float64,
    device: str = "cuda",
    phi_scaled: bool = True,
):
    model = model.to(device).eval()
    w_scale_map = build_w_scale_map_smallcnn_staxlike(model, conv_bias_scale=hy_params["bias_conv"],
                                                      dense_bias_scale=hy_params['bias_dense'])
    X = X.to(device)

    with torch.no_grad():
        C = model(X[:1]).shape[-1]
    if phi_scaled:
        phi_single = make_per_sample_phi_fn_scaled(model, w_scale_map=w_scale_map)
    else:
        phi_single = make_per_sample_phi_fn(model)
    phi_batch = vmap(lambda x: phi_single(x.unsqueeze(0)))(X[:1])  # just to trace shapes

    phi_batch = vmap(lambda x: phi_single(x.unsqueeze(0)))  # maps (1,28,28) -> (C*P,)

    N = X.shape[0]
    out_dev = "cpu" if store_cpu else device

    # accumulate sum in high precision
    phi_sum = torch.zeros((C,), device=out_dev, dtype=dtype)  # placeholder, will resize after first block
    phi_sum = None

    for i0 in range(0, N, block):
        i1 = min(i0 + block, N)
        Phi = phi_batch(X[i0:i1]).to(dtype)  # (bi, C*P)

        if average_over_outputs:
            # If you want sample-mean of per-class Jacobians averaged over outputs,
            # reshape (bi, C, P) -> mean over C -> (bi, P)
            bi = Phi.shape[0]
            Phi = Phi.view(bi, C, -1).mean(dim=1)  # (bi, P)

        # sum over samples in this block
        block_sum = Phi.sum(dim=0)  # (C*P,) or (P,)

        if store_cpu:
            block_sum = block_sum.detach().cpu()
        else:
            block_sum = block_sum.detach()

        if phi_sum is None:
            phi_sum = block_sum.clone()
        else:
            phi_sum += block_sum

    phi_mean = phi_sum / float(N)

    if return_cp and (not average_over_outputs):
        # reshape back to (C, P)
        phi_mean = phi_mean.view(C, -1)

    return phi_mean

def phi_stats(phi_mean: torch.Tensor, name="phi_mean", clip_val=0.1):
    x = phi_mean.detach().flatten().double().cpu()
    n = x.numel()

    q = torch.quantile(x, torch.tensor([0.0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0],
                                       dtype=x.dtype))
    num_clip = int((x == clip_val).sum().item())
    frac_clip = num_clip / n

    print(f"=== Stats: {name} ===")
    print(f"count     : {n}")
    print(f"mean      : {x.mean().item():.6g}")
    print(f"std       : {x.std(unbiased=False).item():.6g}   (ddof=0)")
    print(f"min/max   : {x.min().item():.6g} / {x.max().item():.6g}")
    print(f"median    : {q[4].item():.6g}")
    print("quantiles :")
    labels = ["0%", "1%", "5%", "25%", "50%", "75%", "95%", "99%", "100%"]
    for lab, val in zip(labels, q):
        print(f"  {lab:>4} : {val.item():.6g}")

    print(f"== clip check (== {clip_val}) ==")
    print(f"count_clip: {num_clip}  ({frac_clip:.3%})")

def empirical_ntk_blocks(model, X1, X2,
                         store_cpu=True, ntk_empirical_rescale=False, hy_params=None):
    param = next(model.parameters())
    dtype = param.dtype
    block = hy_params['Nb']; average_over_outputs = hy_params['Class_avg']; device = hy_params['device']
    model = model.to(device).eval()
    X1 = X1.to(device=device, dtype=param.dtype)
    X2 = X2.to(device=device, dtype=param.dtype)

    with torch.no_grad():
        C = model(X1[:1]).shape[-1]

    if ntk_empirical_rescale and not hy_params['use_resnet']:
        w_scale_map = build_w_scale_map_smallcnn_staxlike(model, conv_bias_scale=hy_params["bias_conv"],
                                                          dense_bias_scale=hy_params['bias_dense'])
        phi_single = make_per_sample_phi_fn_scaled(model, w_scale_map=w_scale_map)
    else:
        phi_single = make_per_sample_phi_fn(model)
    phi_batch = vmap(lambda x: phi_single(x.unsqueeze(0)))  # x: (1,28,28) -> add batch dim

    N1, N2 = X1.shape[0], X2.shape[0]
    out_dev = "cpu" if store_cpu else device
    K = torch.empty((N1, N2), device=out_dev, dtype=dtype)

    for i0 in range(0, N1, block):
        i1 = min(i0 + block, N1)

        Phi1 = phi_batch(X1[i0:i1]).to(dtype)  # (bi, C*P)

        for j0 in range(0, N2, block):
            j1 = min(j0 + block, N2)
            Phi2 = phi_batch(X2[j0:j1]).to(dtype)  # (bj, C*P)

            K_block = Phi1 @ Phi2.T  # (bi,bj)
            if average_over_outputs:
                K_block = K_block / C

            if store_cpu:
                K[i0:i1, j0:j1] = K_block.detach().cpu()
            else:
                K[i0:i1, j0:j1] = K_block

            del Phi2, K_block

        del Phi1

    if isinstance(K, torch.Tensor):
        K = K.detach().cpu().numpy().astype(np.float64)
    return K


def empirical_ntk_vjp_jvp_blocks(model, X1, X2, store_cpu=True, ntk_empirical_rescale=False, hy_params=None):
    device = hy_params['device']
    dtype = next(model.parameters()).dtype
    block = hy_params['Nb']
    average_over_outputs = hy_params['Class_avg']
    model = model.to(device).eval()
    if ntk_empirical_rescale:
        w_scale_map = build_w_scale_map_smallcnn_staxlike(model, conv_bias_scale=hy_params["bias_conv"], dense_bias_scale=hy_params['bias_dense'])
    else:
        w_scale_map = None
    params = dict(model.named_parameters())
    buffers = dict(model.named_buffers())

    # 1. Define scaled functional call
    def f_scaled(p, x):
        if ntk_empirical_rescale:
            p_scaled = {n: (v * w_scale_map[n] if n in w_scale_map else v)
                        for n, v in p.items()}
            return functional_call(model, (p_scaled, buffers), (x.unsqueeze(0),)).squeeze(0)
        return functional_call(model, (p, buffers), (x.unsqueeze(0),)).squeeze(0)

    # 2. VJP/JVP Core for a single pair of samples
    def ntk_vp_core(x1, x2):
        out2, vjp_fn = vjp(lambda p: f_scaled(p, x2), params)
        num_classes = out2.shape[-1]
        basis = torch.eye(num_classes, device=device, dtype=dtype)

        def get_column(v):
            vjp_out = vjp_fn(v)[0]
            _, jvp_out = jvp(lambda p: f_scaled(p, x1), (params,), (vjp_out,))
            return jvp_out

        return vmap(get_column)(basis)  # Returns (C, C)

    ntk_batch = vmap(vmap(ntk_vp_core, in_dims=(None, 0)), in_dims=(0, None))

    # 3. Setup output matrix
    N1, N2 = X1.shape[0], X2.shape[0]
    out_dev = "cpu" if store_cpu else device
    K = torch.empty((N1, N2), device=out_dev, dtype=dtype)

    with torch.no_grad():
        C = model(X1[:1].to(device)).shape[-1]

    # 4. Block-wise execution
    for i0 in range(0, N1, block):
        i1 = min(i0 + block, N1)
        x1_block = X1[i0:i1].to(device)

        for j0 in range(0, N2, block):
            j1 = min(j0 + block, N2)
            x2_block = X2[j0:j1].to(device)

            # Compute NTK for this block pair (Shape: bi, bj, C, C)
            K_block = ntk_batch(x1_block, x2_block)

            # Trace over class dimensions if averaging
            if average_over_outputs:
                K_block = torch.einsum('nijj->ni', K_block) / C
            else:
                K_block = torch.einsum('nijj->ni', K_block)

            # Store results
            if store_cpu:
                K[i0:i1, j0:j1] = K_block.detach().cpu()
            else:
                K[i0:i1, j0:j1] = K_block

            del x2_block, K_block

        del x1_block

    return K.cpu().numpy().astype(np.float64)


def empirical_ntk_blocks_update_tr_te(model, X1, y1, X_test, store_cpu=True, ntk_empirical_rescale=True,
                                            hy_params=None):
    param = next(model.parameters())
    dtype = param.dtype

    block = hy_params['Nb']; average_over_outputs = hy_params['Class_avg']; device = hy_params['device']
    X1 = X1.to(device)
    with torch.no_grad():
        C = model(X1[:1]).shape[-1]

    model = model.to(device).eval()
    w_scale_map = build_w_scale_map_smallcnn_staxlike(model, conv_bias_scale=hy_params["bias_conv"], dense_bias_scale=hy_params['bias_dense'])

    if ntk_empirical_rescale:
        phi_single = make_per_sample_phi_fn_scaled(model, w_scale_map=w_scale_map)
    else:
        phi_single = make_per_sample_phi_fn(model)
    phi_batch = vmap(lambda x: phi_single(x.unsqueeze(0)))  # x: (1,28,28) -> add batch dim

    N1, N_test = X1.shape[0], X_test.shape[0]
    Phi_train_list, Phi_test = [], phi_batch(X_test[:N_test]).to(dtype)

    for i0 in range(0, N1, block):
        i1 = min(i0 + block, N1)
        Phi1 = phi_batch(X1[i0:i1]).to(dtype)  # (bi, C*P)
        Phi_train_list.append(Phi1)
        del Phi1
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
        model.train()
        opt = torch.optim.SGD(model.parameters(), lr=hy_params['ntk_update_rate'], momentum=hy_params['sgd_momentum'],
                              weight_decay=hy_params['sgd_weight_decay'])
        criterion = nn.CrossEntropyLoss()
        logits = model(X1[i0:i1])
        loss = criterion(logits, torch.from_numpy(y1[i0:i1]))
        opt.zero_grad(set_to_none=True)  # clear old grads
        loss.backward()  # compute grads dloss/dtheta
        opt.step()
        model.eval()

        if ntk_empirical_rescale:
            phi_single = make_per_sample_phi_fn_scaled(model, w_scale_map=w_scale_map)
        else:
            phi_single = make_per_sample_phi_fn(model)
        phi_batch = vmap(lambda x: phi_single(x.unsqueeze(0)))  # x: (1,28,28) -> add batch dim

    # at the end of empirical_ntk_blocks before return
    Phi_all = torch.cat([phi.to(dtype) for phi in Phi_train_list], dim=0)  # (N, D)
    K_tr = Phi_all @ Phi_all.T  # (N, N)
    K_tr_te = Phi_all @ Phi_test.T
    if average_over_outputs:
        assert C is not None, "Need C (# outputs/classes) to average."
        K_tr = K_tr / C; K_tr_te = K_tr_te / C

    if store_cpu:
        K_tr = K_tr.detach().cpu()
        K_tr_te = K_tr_te.detach().cpu()

    if isinstance(K_tr, torch.Tensor):
        K_tr = K_tr.detach().cpu().numpy().astype(np.float64)
    if isinstance(K_tr_te, torch.Tensor):
        K_tr_te = K_tr_te.detach().cpu().numpy().astype(np.float64)
    return K_tr, K_tr_te

def normalize_kernels(
    K_train: np.ndarray,
    K_trte: np.ndarray,
    *,
    normalize: bool = True,
    method: str = "train_diag",   # "none", "scale", "train_diag", "full_diag"
    eps: float = 1e-12,
    K_test: np.ndarray | None = None,  # only needed for "full_diag"
):
    if (not normalize) or method == "none":
        return K_train, K_trte

    K_train = np.asarray(K_train, dtype=np.float64)
    K_trte  = np.asarray(K_trte,  dtype=np.float64)

    if method == "scale":
        s = float(np.mean(np.diag(K_train)))
        s = max(s, eps)
        return K_train / s, K_trte / s

    # train-side diagonal
    d_tr = np.clip(np.diag(K_train), eps, None)
    inv_tr = 1.0 / np.sqrt(d_tr)

    K_train_n = (K_train * inv_tr[:, None]) * inv_tr[None, :]
    K_trte_n  = (K_trte  * inv_tr[:, None])

    if method == "train_diag":
        return K_train_n, K_trte_n

    if method == "full_diag":
        if K_test is None:
            raise ValueError("method='full_diag' requires K_test (Ntest,Ntest) to get test diagonal.")
        d_te = np.clip(np.diag(np.asarray(K_test, dtype=np.float64)), eps, None)
        inv_te = 1.0 / np.sqrt(d_te)
        K_trte_n = K_trte_n * inv_te[None, :]
        return K_train_n, K_trte_n

    raise ValueError(f"Unknown method={method}")


def KU_to_KbU(KU: np.ndarray, block_size: int) -> np.ndarray:
    KU = np.asarray(KU)
    assert KU.ndim == 2 and KU.shape[0] == KU.shape[1], "KU must be (n,n)"
    n = KU.shape[0]
    b = int(block_size)
    assert b > 0, "block_size must be positive"

    blk = np.arange(n) // b                  # block index per row/col
    mask = (blk[:, None] < blk[None, :])     # keep only blocks strictly above diagonal
    return KU * mask                         # elementwise mask


def feature_kernel(model, x1, x2, hy_params, device='cuda'):
    model.eval()
    model.to(device)

    with torch.no_grad():
        C = model(x1[:1].to(device)).shape[-1]

    average_over_outputs = hy_params.get('Class_avg', False)

    # 1. Define the feature extractor block
    feature_extractor = torch.nn.Sequential(
        model.conv1, model.bn1, model.relu, model.maxpool,
        model.layer1, model.layer2, model.layer3, model.layer4,
        model.avgpool,
        torch.nn.Flatten(1)
    )

    # 2. Compute features for both batches
    with torch.no_grad():
        x1_feats = feature_extractor(x1.to(device))  # Shape: (N1, 512)
        x2_feats = feature_extractor(x2.to(device))  # Shape: (N2, 512)

    # 3. Compute the full pairwise dot product Gram Matrix K
    K = torch.matmul(x1_feats, x2_feats.T)

    # --- NEW: Average over outputs if requested ---
    if average_over_outputs:
        K = K / C

    return K