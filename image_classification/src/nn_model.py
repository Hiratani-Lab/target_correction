import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from neural_tangents import stax
from dataset import torch_to_jax_nhwc
from kernel_cal import KU_to_KbU, empirical_ntk_blocks_classwise2
from ewc_gating import infer_logits_optionA, ClassGatedSmallCNN, TaskGatedCNN, infer_logits_optionB, infer_pred_optionA_selfclass, infer_logits_entropy_gate
from ewc_gating import ewc_penalty, estimate_fisher_diag, snapshot_params
from kernel_cal import (compute_ntk_features, kernel_from_features, calc_kernels_jax, empirical_ntk_blocks,
                        empirical_ntk_blocks_classwise, normalize_kernels, empirical_ntk_vjp_jvp_blocks, feature_kernel)

def one_hot(y: np.ndarray, C=10) -> np.ndarray:
    y = np.asarray(y, dtype=int)
    oh = np.zeros((len(y), C), dtype=np.float64)
    oh[np.arange(len(y)), y] = 1.0
    return oh

_TASK_INCRE_FULL_TEST_EVAL_DEFAULT = True

def _task_incre_use_full_test_eval(hy_params) -> bool:
    return bool(hy_params.get("task_incre_full_test_eval", _TASK_INCRE_FULL_TEST_EVAL_DEFAULT))


def incre_seen_test_subset(y_train_seen, X_test, y_test):
    # 1. Identify which classes are in the current training subset
    # y_train[:N] contains the labels the model has encountered so far
    seen_classes = np.unique(y_train_seen)

    # 2. Find indices in the test set that belong to these seen classes
    # This creates a boolean mask (True for seen classes, False otherwise)
    test_mask = np.isin(y_test, seen_classes)

    # 3. Create the filtered test subset
    X_test_subset = X_test[test_mask]
    y_test_subset = y_test[test_mask]

    return X_test_subset, y_test_subset


def incre_seen_test_subset_task_incremental(X_test_seq, y_test_seq, current_task_idx, total_tasks=5):
    # 1. Calculate exactly how many test samples belong to a single task
    # If len(X_test_seq) is 10,000, then samples_per_task is exactly 2,000
    samples_per_task = len(X_test_seq) // total_tasks

    # 2. Calculate the slice cutoff index
    # If current_task_idx is 0 (Task 1), cutoff is 2000.
    # If current_task_idx is 1 (Task 2), cutoff is 4000, and so on.
    cutoff_idx = (current_task_idx + 1) * samples_per_task

    # 3. Slice the arrays from the beginning up to the cutoff
    X_test_subset = X_test_seq[:cutoff_idx]
    y_test_subset = y_test_seq[:cutoff_idx]

    return X_test_subset, y_test_subset


def build_resnet18_model(device, num_classes=50, use2dense=False, hidden_dim=100, bias_flag=True, update_cnn=False):
    # Load pre-trained ResNet18
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    # 1. Freeze CNN layers
    for param in model.parameters():
        param.requires_grad = update_cnn

    # 2. Replace classification head
    num_ftrs = model.fc.in_features

    if use2dense:
        # Add the hidden layer and its nonlinearity, passing the bias_flag
        model.fc = nn.Sequential(
            nn.Linear(num_ftrs, hidden_dim, bias=bias_flag),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_classes, bias=bias_flag)
        )
    else:
        # Map directly from flattened features to output classes, passing the bias_flag
        model.fc = nn.Linear(num_ftrs, num_classes, bias=bias_flag)

    # 3. Ensure the newly created classification head requires gradients
    for param in model.fc.parameters():
        param.requires_grad = True

    return model.to(device)


class SmallCNN(nn.Module):
    def __init__(self, width=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, width, 3, padding=1), nn.ReLU(),
            nn.Conv2d(width, width, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),  # 14x14
            nn.Conv2d(width, 2 * width, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),  # 7x7
            nn.Flatten(),
            nn.Linear(2 * width * 7 * 7, 128), nn.ReLU(),
            nn.Linear(128, 10, bias=False)   # logits
        )

    def forward(self, x):
        return self.net(x)

class SmallCNN_StaxLike(nn.Module):
    def __init__(self, n_channels=1, num_classes=10,
                 hidden_dim=1024, bias_conv=True, bias_dense=True, use2dense=True):
        super().__init__()

        # 1. Define the Feature Extractor (Convolutional Layers)
        layers = [
            nn.Conv2d(n_channels, 32, kernel_size=3, padding=1, bias=bias_conv),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=2, stride=2),  # 28 -> 14

            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=bias_conv),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((7, 7)),  # 14 -> 7

            nn.Flatten()
        ]

        # 2. Add the Classifier (Dense Layers)
        flattened_size = 64 * 7 * 7

        if use2dense:
            # Add the hidden layer and its nonlinearity
            layers.extend([
                nn.Linear(flattened_size, hidden_dim, bias=bias_dense),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, num_classes, bias=bias_dense)
            ])
        else:
            # Map directly from flattened features to output classes
            layers.append(
                nn.Linear(flattened_size, num_classes, bias=bias_dense)
            )

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class SmallVGG_CIFAR10(nn.Module):
    def __init__(
        self,
        n_channels=3,
        num_classes=10,
        bias_conv=True,
        bias_dense=True,
        norm="group"
    ):
        super().__init__()

        def Norm(c):
            if norm == "batch":
                return nn.BatchNorm2d(c)
            elif norm == "group":
                return nn.GroupNorm(num_groups=8, num_channels=c)
            else:
                return nn.Identity()

        self.features = nn.Sequential(
            nn.Conv2d(n_channels, 64, 3, padding=1, bias=bias_conv),
            Norm(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 64, 3, padding=1, bias=bias_conv),
            Norm(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1, bias=bias_conv),
            Norm(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 128, 3, padding=1, bias=bias_conv),
            Norm(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, 3, padding=1, bias=bias_conv),
            Norm(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 256, bias=bias_dense),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes, bias=bias_dense),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

def SmallCNN_model(hy_params_vanilla, use_gate):
    if hy_params_vanilla['train_order_type']=='task_incremental':
        num_classes=2
        num_tasks=5
    elif hy_params_vanilla['train_order_type']=='random' or hy_params_vanilla['train_order_type']=='class_incremental':
        num_classes=10
        num_tasks=1
    else:
        num_classes=0
        print("incorrect train order type")

    if hy_params_vanilla['ds_type'] == 'mnist':
        n_channels=1
    elif hy_params_vanilla['ds_type'] == 'cifar10':
        n_channels=3
    else:
        n_channels=None
    if not use_gate:
        model_ntk = SmallCNN_StaxLike(n_channels=n_channels, num_classes=num_classes, bias_conv=hy_params_vanilla['bias_conv'],
                                      bias_dense=hy_params_vanilla['bias_dense'], use2dense=hy_params_vanilla['use2dense'])
    else:
        if hy_params_vanilla['train_order_type'] == 'task_incremental':
            model_ntk = TaskGatedCNN(n_channels=n_channels, num_classes=num_classes,
                                    num_tasks=num_tasks,
                                    gate_frac=hy_params_vanilla['gate_frac'], bias_conv=hy_params_vanilla['bias_conv'],
                                    bias_dense=hy_params_vanilla['bias_dense'], use_gating=True, use2dense=hy_params_vanilla['use2dense'])
        else:
            model_ntk = ClassGatedSmallCNN(n_channels=n_channels, num_classes=num_classes, gate_frac=hy_params_vanilla['gate_frac'], bias_conv=hy_params_vanilla['bias_conv'],
                                           bias_dense=hy_params_vanilla['bias_dense'], use_gating=True, use2dense=hy_params_vanilla['use2dense'])
    return model_ntk

import inspect, torch.nn as nn

def SmallConvNetJax(num_classes=10, bias_conv=1., bias_dense=1.):
	return stax.serial(
		stax.Conv(32, (3,3), padding='SAME', W_std=1.0, b_std=bias_conv),
		stax.Relu(), stax.AvgPool((2,2), strides=(2, 2)),
		stax.Conv(64, (3,3), padding='SAME', W_std=1.0, b_std=bias_conv),
		stax.Relu(), stax.AvgPool((2,2), strides=(2, 2)),
		stax.Flatten(),
		stax.Dense(num_classes, 1., b_std=bias_dense))


def krr_predict_precomputed(K_dd: np.ndarray, K_dq: np.ndarray, Yd: np.ndarray, gamma: float):
    d = K_dd.shape[0]
    A = K_dd + gamma * np.eye(d, dtype=np.float64)
    W = np.linalg.solve(A, K_dq)   # (d,m)
    f = Yd.T @ W                   # (10,m)
    pred = np.argmax(f, axis=0)
    return f, pred


def krr_predict_classwise(K_train_c: np.ndarray,
                         K_test_c: np.ndarray,
                         Y_train_oh: np.ndarray,
                         gamma: float):
    C, N, _ = K_train_c.shape
    _, N2, M = K_test_c.shape
    assert N2 == N
    assert Y_train_oh.shape == (N, C)

    F = np.zeros((M, C), dtype=np.float64)
    I = np.eye(N, dtype=np.float64)

    for c in range(C):
        A = K_train_c[c] + gamma * I
        y = Y_train_oh[:, c]                  # (N,)
        alpha = np.linalg.solve(A, K_test_c[c])  # (N, M)
        f_c = (y.T @ alpha).T         # [(,N) @ (N, M)].T = (M, )
        F[:, c] = f_c

    pred = F.argmax(axis=1)
    return F, pred

def on_predict_eq1_precomputed(K_dd, K_dq, Zd, eta, block_size=1):
    d = K_dd.shape[0]
    Ku = np.triu(K_dd, k=1)
    if block_size>1:
        Kbu = KU_to_KbU(Ku, block_size)
        A = (1.0 / eta) * np.eye(d, dtype=np.float64) + Kbu
    else:
        A = (1.0 / eta) * np.eye(d, dtype=np.float64) + Ku

    cond = np.linalg.cond(A)
    if not np.isfinite(cond) or cond > 1e12:
        print("lstsq used in calculating W")
        W = np.linalg.lstsq(A + 1e-6 * np.eye(d), K_dq, rcond=None)[0]
    else:
        W = np.linalg.solve(A, K_dq)
    if not np.all(np.isfinite(W)):
        print("W has nan/inf", np.nanmin(W), np.nanmax(W))
    f = Zd.T @ W
    pred = np.argmax(f, axis=0)
    return f, pred

def on_predict_classwise(K_train_c: np.ndarray,
                         K_test_c: np.ndarray,
                         Z_train_oh: np.ndarray,
                         eta: float,
                         block_size=1):
    C, N, _ = K_train_c.shape
    _, N2, M = K_test_c.shape
    assert N2 == N
    assert Z_train_oh.shape == (N, C)

    F = np.zeros((M, C), dtype=np.float64)
    I = np.eye(N, dtype=np.float64)

    for c in range(C):
        # print("K_train shape", K_train_c.shape)
        KU = np.triu(K_train_c[c], k=1)
        if block_size > 1:
            Kbu = KU_to_KbU(KU, block_size)
            A = (1.0 / eta) * I + Kbu
        else:
            A = (1.0 / eta) * I + KU
        cond = np.linalg.cond(A)
        if not np.isfinite(cond) or cond > 1e12:
            print("lstsq used in calculating W")
            alpha = np.linalg.lstsq(A + 1e-6 * I, K_test_c[c], rcond=None)[0]
        else:
            alpha = np.linalg.solve(A, K_test_c[c])

        z = Z_train_oh[:, c]                  # (N,)
        f_c = (z.T @ alpha).T         # [(,N) @ (N, M)].T = (M, )
        F[:, c] = f_c

    pred = F.argmax(axis=1)
    return F, pred


def KbU_mini_batch(K, batch_size):
    N = K.shape[0]

    if isinstance(K, torch.Tensor):
        # Assign block index, keeping operations on the exact same device as K
        block_idx = torch.arange(N, device=K.device) // batch_size

        # Create mask: row_block < col_block
        mask = block_idx.unsqueeze(1) < block_idx.unsqueeze(0)

        # Apply mask and return a PyTorch Tensor
        return torch.where(mask, K, torch.zeros_like(K))

    elif isinstance(K, np.ndarray):
        # Assign block index
        block_idx = np.arange(N) // batch_size

        # Create mask: row_block < col_block (using None for dimensionality)
        mask = block_idx[:, None] < block_idx[None, :]

        # Assuming K is float, 0.0 is used to maintain float dtype
        return np.where(mask, K, 0.0)

    else:
        raise TypeError(f"Input K must be a torch.Tensor or numpy.ndarray. Got {type(K)}.")


def compute_corrected_Z_iter(K_train: np.ndarray, Y_train_oh: np.ndarray, gamma: float, eta: float, eps: float = 1e-12, z_type: str = "both", mini_bz: int = 1):
    n = K_train.shape[0]
    Z = np.zeros((n, 10), dtype=np.float64)
    Z[0] = Y_train_oh[0]  # z0 = y0

    for d in range(1, n):
        K_prev = K_train[:d, :d]           # (d,d)
        k_prev_d = K_train[:d, d].copy()   # (d,)
        k_dd = float(K_train[d, d])        # scalar

        Y_prev = Y_train_oh[:d]            # (d,10)
        Z_prev = Z[:d]                     # (d,10)
        y_d = Y_train_oh[d]                # (10,)

        A_kr = K_prev + gamma * np.eye(d, dtype=np.float64)
        v = np.linalg.solve(A_kr, k_prev_d)      # (d,)
        f_kr_prev = (Y_prev.T @ v)               # (10,)
        q = float(k_prev_d @ v)                  # scalar

        # f_on^{d-1}(x_d) with K^U
        Ku_prev = KbU_mini_batch(K_prev, mini_bz)
        A_on = (1.0 / eta) * np.eye(d, dtype=np.float64) + Ku_prev
        cond = np.linalg.cond(A_on)
        if not np.isfinite(cond) or cond > 1e12:
            print("lstsq used in calculating u")
            u = np.linalg.lstsq(A_on + 1e-6 * np.eye(d), k_prev_d, rcond=None)[0]
        else:
            u = np.linalg.solve(A_on, k_prev_d)

        if not np.all(np.isfinite(u)):
            print("u has nan/inf", np.nanmin(u), np.nanmax(u))

        f_on_prev = (Z_prev.T @ u)               # (10,)

        a = (1.0 / (eta * (k_dd + eps)) - 1.0)   # scalar
        denom = (eta * k_dd) * (gamma + k_dd - q)
        denom = denom if abs(denom) > eps else (eps if denom >= 0 else -eps)
        b = gamma / denom                         # scalar

        if z_type=='both':
            Z[d] = y_d + a * (y_d - f_on_prev) + b * (f_kr_prev - y_d)
        elif z_type=='online':
            Z[d] = y_d + a * (y_d - f_on_prev)
        else:
            Z[d] = y_d + b * (f_kr_prev - y_d)
    return Z

@torch.no_grad()
def compute_corrected_Z_batch_gpu(K_train: torch.Tensor, Y_train_oh: torch.Tensor, gamma: float, eta: float, batch_size: int = 128,
                                  eps: float = 1e-12, z_type: str = "both", device: str = "cuda", mini_bz: int = 1):
    K = torch.as_tensor(K_train, dtype=torch.float64, device=device)
    Y = torch.as_tensor(Y_train_oh, dtype=torch.float64, device=device)

    n, c = K.shape[0], Y.shape[1]
    Z = torch.zeros((n, c), dtype=torch.float64, device=device)
    _lstsq_jitter = 1e-6

    def _solve_or_lstsq(A: torch.Tensor, B: torch.Tensor, batch_d: int, name: str) -> torch.Tensor:
        """Prefer LU solve; fall back to damped lstsq if A is singular / ill-conditioned."""
        try:
            return torch.linalg.solve(A, B)
        except torch._C._LinAlgError:
            dd = A.shape[0]
            print(f"lstsq in cal {name}")
            A_stab = A + _lstsq_jitter * torch.eye(dd, dtype=torch.float64, device=device)
            return torch.linalg.lstsq(A_stab, B).solution

    # Initialize the first batch
    m_init = min(batch_size, n)
    Z[:m_init] = Y[:m_init]
    for d in range(m_init, n, batch_size):
        end = min(d + batch_size, n)
        m = end - d

        # 1. Extract Block Matrices
        K_prev = K[:d, :d]
        K_cross = K[:d, d:end]
        K_m = K[d:end, d:end]

        Y_prev = Y[:d]
        Z_prev = Z[:d]
        Y_m = Y[d:end]
        # ==========================================
        # 2. Offline Predictions & Q_m Matrix
        # ==========================================
        A_kr = K_prev.clone()
        A_kr.diagonal().add_(gamma)

        V_kr = _solve_or_lstsq(A_kr, K_cross, d, "V_kr")
        f_kr_m = V_kr.T @ Y_prev
        Q_m = K_cross.T @ V_kr
        # ==========================================
        # 3. Online Predictions (with try-except speedup)
        # ==========================================
        # torch.triu allocates a new tensor, so we can safely modify its diagonal in-place
        A_on = KbU_mini_batch(K_prev, mini_bz)

        # OPTIMIZATION: Add to diagonal in-place (Saves creating torch.eye)
        A_on.diagonal().add_(1.0 / eta)

        V_on = _solve_or_lstsq(A_on, K_cross, d, "V_on")

        f_on_m = V_on.T @ Z_prev

        # ==========================================
        # 4. Compute H and P_m Matrices
        # ==========================================
        K_m_U = KbU_mini_batch(K_m, mini_bz)
        H_right = (1.0 / eta) * torch.eye(m, dtype=torch.float64, device=device) + K_m_U

        # Solve for H (K_m is small, BxB, so this is fast)
        stable_K_m = K_m + eps * torch.eye(m, dtype=torch.float64, device=device)
        H = _solve_or_lstsq(stable_K_m, H_right, d, "H")

        P_m_inv = gamma * torch.eye(m, dtype=torch.float64, device=device) + K_m - Q_m
        I_m = torch.eye(m, dtype=torch.float64, device=device)
        # P_m = P_m_inv^{-1}; inv() is less stable than solve(P_m_inv, I)
        P_m = _solve_or_lstsq(P_m_inv, I_m, d, "P_m")

        # ==========================================
        # 5. Final Batch Label Correction (Z_m)
        # ==========================================
        M_on = H.T - torch.eye(m, dtype=torch.float64, device=device)
        M_off = gamma * (H.T @ P_m)

        if z_type == 'both':
            Z_m = Y_m + M_on @ (Y_m - f_on_m) + M_off @ (f_kr_m - Y_m)
        elif z_type == 'online':
            Z_m = Y_m + M_on @ (Y_m - f_on_m)
        elif z_type == 'offline':
            Z_m = Y_m + M_off @ (f_kr_m - Y_m)

        Z[d:end] = Z_m

        # Delete large intermediate matrices before the next loop starts
        del K_prev, K_cross, A_kr, V_kr, A_on, V_on, Q_m

    # Return Z to CPU as a NumPy array to match your previous pipeline
    return Z.cpu().numpy()


def compute_corrected_Z_full(K_train, Y_train_oh, gamma, eta, jitter=1e-8, use_cholesky=False, mini_bz=1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 1. Transfer to GPU
    K = torch.as_tensor(K_train, device=device, dtype=torch.float32)
    Y = torch.as_tensor(Y_train_oh, device=device, dtype=torch.float32)
    n = K.shape[0]
    I = torch.eye(n, device=device, dtype=torch.float32)

    # 2. Build the left and right matrices
    # left = (gamma * I + K + jitter * I)
    left = (gamma + jitter) * I + K

    KU = KbU_mini_batch(K, mini_bz)
    right = (1.0 / eta) * I + KU

    Begin_solve=time.time()
    # 3. Solve AX = B
    if use_cholesky:
        try:
            L = torch.linalg.cholesky(left)
            A = torch.linalg.cholesky_solve(L, right)
        except RuntimeError:
            try:
                # 1. Attempt the fast, standard solver first
                A = torch.linalg.solve(left, right)
            except torch._C._LinAlgError:
                # 2. If singular/ill-conditioned, use the robust Least Squares solver
                # This uses SVD or QR decomposition under the hood
                A = torch.linalg.lstsq(left, right).solution
    else:
        # Standard general-purpose solver
        try:
            # 1. Attempt the fast, standard solver first
            A = torch.linalg.solve(left, right)
        except torch._C._LinAlgError:
            print("using lstsq to solve the inverse matrix")
            # 2. If singular/ill-conditioned, use the robust Least Squares solver
            # This uses SVD or QR decomposition under the hood
            A = torch.linalg.lstsq(left, right).solution

    End_solve=time.time()
    print("time cost when solve the inverse matrix:", End_solve-Begin_solve)

    # 4. Final calculation: Z = A.T @ Y
    Z = A.T @ Y

    # Convert the GPU tensor to a NumPy array first
    z_numpy = Z.cpu().numpy()

    # Use astype to match the input's original precision (e.g., float64)
    # copy=False ensures we don't duplicate memory if the types already match
    return z_numpy.astype(Y_train_oh.dtype, copy=False)

def calc_iter_labelZ_correction(Kdata, Ytrain, Niters, hy_params):
    eta, gm, mini_bz = hy_params['eta'], hy_params['gm'], hy_params['sgd_batch_size']
    Yc_iter = np.zeros((np.shape(Ytrain)))

    dN = Niters[1] - Niters[0]
    IdN = np.eye(dN)
    for Nidx, N in enumerate(Niters):
        if Nidx == 0:
            Yc_iter[:N, :] = Ytrain[:N, :]
        else:
            yon_new, pred_label = on_predict_eq1_precomputed(Kdata[:N - dN, :N - dN], Kdata[:N - dN, N - dN:N], Yc_iter[:N - dN, :], eta)
            dyon_new = Ytrain[N - dN:N, :] - yon_new.T # (dN, 10)

            Moff_new = gm * IdN + Kdata[N - dN:N, N - dN:N]
            Mon_new = (1.0 / eta) * IdN + KbU_mini_batch(Kdata[N - dN:N, N - dN:N], mini_bz)
            Cnew = np.linalg.solve(Moff_new, Mon_new) - IdN

            Yc_iter[N - dN:N, :] = Ytrain[N - dN:N, :] + np.dot(Cnew.T, dyon_new)

    return Yc_iter

def calc_iter_labelZ_correction_eq46(Kdata, Ytrain, Niters, hy_params):
    eta, gm, mini_bz = hy_params['eta'], hy_params['gm'], hy_params['sgd_batch_size']

    Yc_iter = np.zeros((np.shape(Ytrain)))
    dN = Niters[1] - Niters[0]
    IdN = np.eye(dN)
    for Nidx, N in enumerate(Niters):
        if Nidx == 0:
            Yc_iter[:N, :] = Ytrain[:N, :]
        else:
            yon_new, pred_label = on_predict_eq1_precomputed(Kdata[:N - dN, :N - dN], Kdata[:N - dN, N - dN:N], Yc_iter[:N - dN, :], eta)
            dyon_new = Ytrain[N - dN:N, :] - yon_new.T # (dN, 10)

            Moff_new = gm * IdN + Kdata[N - dN:N, N - dN:N]
            Mon_new = (1.0 / eta) * IdN + KbU_mini_batch(Kdata[N - dN:N, N - dN:N], mini_bz)
            Cnew = np.linalg.solve(Moff_new, Mon_new)

            Yc_iter[N - dN:N, :] = np.dot(Cnew.T, dyon_new)

    return Yc_iter

def mse_loss(logits, y_onehot):
    return (0.5 * (logits - y_onehot) ** 2).sum(dim=1).mean()

@torch.no_grad()
def accuracy_cnn(model, X, y, device="cpu", batch_size=256, gate_mode="none", true_gate_ids=None):
    device = torch.device(device)
    model.eval()
    # Convert once (handles numpy OR torch inputs)
    X = torch.as_tensor(X, device=device, dtype=torch.float32)
    y_t = torch.as_tensor(y, device=device, dtype=torch.long)

    correct = torch.zeros((), device=device, dtype=torch.long)
    total_mse = torch.zeros((), device=device, dtype=torch.float32)
    n = X.shape[0]

    gating_on = getattr(model, "use_gating", False)
    if true_gate_ids is not None:
        true_gate_ids = torch.as_tensor(true_gate_ids, device=device, dtype=torch.long)

    for i in range(0, n, batch_size):
        xb = X[i:i + batch_size].to(device)
        yb = y_t[i:i + batch_size]

        if true_gate_ids is not None:
            g_ids_batch = true_gate_ids[i:i + batch_size]
        else:
            g_ids_batch = yb

        # ------------- forward by mode -------------
        if gate_mode == "oracle" and gating_on:
            logits = model(xb, gate_ids=g_ids_batch)
            pred = logits.argmax(dim=-1)

        elif gate_mode == "inferA" and gating_on:
            logits, _ = infer_logits_optionA(model, xb)
            pred = logits.argmax(dim=-1)

        else:
            logits = model(xb, gate_ids=None) if ("gate_ids" in model.forward.__code__.co_varnames) else model(xb)
            pred = logits.argmax(dim=-1)

        # ---------------- metrics ----------------
        correct += (pred == yb).sum()

        yb_onehot = F.one_hot(yb, num_classes=logits.shape[1]).to(dtype=logits.dtype)
        mse = mse_loss(logits, yb_onehot)   # uses your existing mse_loss
        total_mse += mse * xb.size(0)
    # after loop
    correct = correct.item()
    total_mse = total_mse.item()
    return correct / n, total_mse / n

def classwise_label_corrected(y_true, y_pred, K_nn_c, eta, gm, Kbu_block_size, mini_bz=1):
    C, dN, _ = K_nn_c.shape
    assert y_true.shape == (dN, C)
    assert y_pred.shape == (dN, C)

    Z = np.zeros((dN, C), dtype=np.float64)
    I = np.eye(dN, dtype=np.float64)

    for c in range(C):
        K = K_nn_c[c]
        if "torch" in str(type(K)):
            K = K.detach().cpu().numpy()
        K = np.asarray(K, dtype=np.float64)

        KU = KbU_mini_batch(K, mini_bz)
        Moff_new = gm * I + K
        if Kbu_block_size > 1:
            Kbu = KU_to_KbU(KU, Kbu_block_size)
            Mon_new = (1.0 / eta) * I + Kbu
        else:
            Mon_new = (1.0 / eta) * I + KU

        try:
            Cnew = np.linalg.solve(Moff_new, Mon_new)  # [dN, dN]
        except np.linalg.LinAlgError:
            print("lstsq used in calculating W")
            Cnew = np.linalg.lstsq(Moff_new + 1e-6 * I, Mon_new, rcond=None)[0]


        dy_np_c = y_true[:, c] - y_pred[:, c]  # (dN, 1)
        Z_c = Cnew.T @ dy_np_c  # (dN, )
        Z[:, c] = Z_c # (dN, c)
    return Z


def kernel_type(model_ntk, X_train, X_test, device, ntk_type, hy_params):
    if ntk_type == "auto_grad":
        print("Computing NTK features (train)...")
        G_train = compute_ntk_features(model_ntk, X_train, device=device, print_every=50)
        print("Computing NTK features (test)...")
        G_test = compute_ntk_features(model_ntk, X_test, device=device, print_every=50)
        K_train = kernel_from_features(G_train, G_train, C=10, Class_avg=hy_params['Class_avg']) # (n_train,n_train)
        K_tr_te = kernel_from_features(G_train, G_test, C=10, Class_avg=hy_params['Class_avg']) # (n_train,n_test)
        return K_train, K_tr_te
    elif ntk_type == "stax_kernel":
        # ----- (B) NTK kernels from stax kernel_fn
        # Use its own freshly initialized model (random features baseline)
        init_fn, apply_fn, kernel_fn = SmallConvNetJax(bias_conv=hy_params['stax_bias_conv'], bias_dense=hy_params['stax_bias_dense'])
        # X_train, X_test are torch tensors
        Xtrain_jax = torch_to_jax_nhwc(X_train)
        Xtest_jax = torch_to_jax_nhwc(X_test)
        print(Xtrain_jax.shape)
        # kernel_fn from stax network
        K_train, K_tr_te = calc_kernels_jax(kernel_fn, Xtrain_jax, Xtest_jax, hy_params)
        return K_train, K_tr_te
    elif ntk_type == "jacobian_empirical_no_rescale":
        # ----- (C) NTK kernels from jacobian, average by class number
        # X_train_all: (Ntrain,1,28,28), X_test: (Ntest,1,28,28)
        K_train = empirical_ntk_blocks(model_ntk, X_train, X_train, ntk_empirical_rescale=False, hy_params=hy_params)
        K_tr_te = empirical_ntk_blocks(model_ntk, X_train, X_test, ntk_empirical_rescale=False, hy_params=hy_params)
        return K_train, K_tr_te
    elif ntk_type == "vjp_jvp_no_rescale":
        K_train = empirical_ntk_vjp_jvp_blocks(model_ntk, X_train, X_train, ntk_empirical_rescale=False, hy_params=hy_params)
        K_tr_te = empirical_ntk_vjp_jvp_blocks(model_ntk, X_train, X_test, ntk_empirical_rescale=False, hy_params=hy_params)
        return K_train, K_tr_te
    elif ntk_type == "jacobian_empirical_rescale":
        # ----- (C) NTK kernels from jacobian, average by class number
        # X_train_all: (Ntrain,1,28,28), X_test: (Ntest,1,28,28)
        K_train = empirical_ntk_blocks(model_ntk, X_train, X_train, ntk_empirical_rescale=True, hy_params=hy_params)
        K_tr_te = empirical_ntk_blocks(model_ntk, X_train, X_test, ntk_empirical_rescale=True, hy_params=hy_params)
        return K_train, K_tr_te
    elif ntk_type == "jacobian_empirical_rescale_diag_norm":
        # ----- (D) NTK kernels from jacobian, train_diag normalizaed, average by class number
        K_train = empirical_ntk_blocks(model_ntk, X_train, X_train, ntk_empirical_rescale=True, hy_params=hy_params)
        K_tr_te = empirical_ntk_blocks(model_ntk, X_train, X_test, ntk_empirical_rescale=True, hy_params=hy_params)
        K_train_diag_n, K_tr_te_diag_n = normalize_kernels(
            K_train, K_tr_te,
            normalize=True,
            method="train_diag",
            eps=1e-12
        )
        return K_train_diag_n, K_tr_te_diag_n
    elif ntk_type == 'jacobian_classwise_no_rescale':
        K_train = empirical_ntk_blocks_classwise(model_ntk, X_train, X_train, hy_params=hy_params)
        K_tr_te = empirical_ntk_blocks_classwise(model_ntk, X_train, X_test, hy_params=hy_params)
        return K_train, K_tr_te
    elif ntk_type == 'feature_kernel':
        K_train = feature_kernel(model_ntk, X_train, X_train, hy_params, device=hy_params['device'])
        K_tr_te = feature_kernel(model_ntk, X_train, X_test, hy_params, device=hy_params['device'])
        return K_train, K_tr_te
    else:
        print("incorrect kernel type")
        return None, None

def sgd_train_streaming_MSE(
    model: nn.Module,
    X_train: torch.Tensor, y_train: np.ndarray,
    X_test: torch.Tensor,  y_test: np.ndarray,
    steps,
    device="cpu",
    lr=0.05,
    momentum=0.0,
    weight_decay=0.0,
    batch_size=32,
):
    model.to(device)
    model.train()
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)

    steps = list(steps)
    step_set = set(steps)
    max_step = max(steps)

    train_accs, test_accs = [], []
    buf_x, buf_y = [], []

    # infer num classes once (works for your SmallCNN_StaxLike)
    with torch.no_grad():
        C = model(X_train[:1].to(device)).shape[-1]

    def flush():
        if not buf_x:
            return
        xb = torch.stack(buf_x, dim=0).to(device)  # (B,1,28,28)
        yb = torch.tensor(buf_y, dtype=torch.long, device=device)  # (B,)

        opt.zero_grad(set_to_none=True)
        logits = model(xb)  # (B, C)

        # one-hot targets for MSE on logits
        y_onehot = F.one_hot(yb, num_classes=C).to(dtype=logits.dtype)  # (B, C)

        loss = mse_loss(logits, y_onehot)
        loss.backward()
        opt.step()

        buf_x.clear()
        buf_y.clear()

    for i in range(max_step):
        buf_x.append(X_train[i])
        buf_y.append(int(y_train[i]))
        if len(buf_x) >= batch_size:
            flush()

        prefix = i + 1
        if prefix in step_set:
            flush()
            tr_acc, tr_mse = accuracy_cnn(model, X_train[:prefix], y_train[:prefix], device=device)
            te_acc, te_mse = accuracy_cnn(model, X_test, y_test, device=device)
            train_accs.append(tr_acc)
            test_accs.append(te_acc)

    return np.array(train_accs), np.array(test_accs)

def sgd_train_streaming_MSE_label_corrected(model, X_train, y_train, X_test, y_test, Niters, hy_params, label_corrected=True):
    eta, gm, KbU_block_size = hy_params['eta'], hy_params['gm'], hy_params['KbU_block_size']

    if hy_params['train_order_type'] == 'task_incremental':
        Y_train_oh = one_hot(y_train, 2)
    else:
        Y_train_oh = one_hot(y_train, 10) # F.one_hot(y, num_classes=10).to(dtype=X_train.dtype)
    Yc_iter_oh = np.zeros((np.shape(Y_train_oh)))
    dN = Niters[1] - Niters[0]
    IdN = np.eye(dN)

    train_accs, test_accs = [], []
    train_mses, test_mses = [], []


    for Nidx, N in enumerate(Niters):
        if Nidx == 0:
            Yc_iter_oh[:N, :] = Y_train_oh[:N, :]
        else:
            if label_corrected:
                X_next, Y_next = X_train[N - dN:N], Y_train_oh[N - dN:N]

                K_nn_c = empirical_ntk_blocks_classwise2(model, X_next, X_next, hy_params=hy_params)
                if N==Niters[1]:
                    print("first classwise kernel matrix for label correction:")
                elif N==Niters[-1]:
                    print("last classwise kernel matrix for label correction:")
                if hy_params['kernel_tp'] == "jacobian_empirical_rescale":
                    K_nn = empirical_ntk_blocks(model, X_next, X_next, store_cpu=True, dtype=torch.float64, ntk_empirical_rescale=True, hy_params=hy_params)
                else:
                    K_nn = K_nn_c.mean(dim=0)
                    if N==Niters[1]:
                        print("first no-classwise kernel matrix for label correction:")
                    elif N==Niters[-1]:
                        print("last no-classwise kernel matrix for label correction:")
                if "torch" in str(type(K_nn)):
                    K_nn = K_nn.detach().cpu().numpy()
                K_nn = np.asarray(K_nn, dtype=np.float64)

                with torch.no_grad():
                    Y_logits_next = model(X_next)  # torch, no grad

                ylog_np = Y_logits_next.detach().cpu().numpy()  # (dN, C), y_pred
                Yb_np = Y_train_oh[N - dN:N, :]  # numpy, (dN, C) y_true

                if hy_params['kernel_tp'] == 'jacobian_classwise_no_rescale':
                    Z = classwise_label_corrected(Yb_np, ylog_np, K_nn_c, eta, gm, KbU_block_size)
                    Yc_iter_oh[N - dN:N, :] = Z # (dN, C)
                else:
                    dy_np = Yb_np - ylog_np  # (dN, C)
                    Moff_new = gm * IdN + K_nn
                    mini_bz = hy_params['sgd_batch_size']
                    Knn_U = KbU_mini_batch(K_nn, mini_bz)
                    if KbU_block_size > 1:
                        Knn_bU = KU_to_KbU(Knn_U, KbU_block_size)
                        Mon_new = (1.0 / eta) * IdN + Knn_bU
                    else:
                        Mon_new = (1.0 / eta) * IdN + Knn_U
                    Cnew = np.linalg.solve(Moff_new, Mon_new) # - IdN
                    Yc_iter_oh[N - dN:N, :] = Cnew.T @ dy_np   # (dN, C)
            else:
                Yc_iter_oh[N - dN:N, :] = Y_train_oh[N - dN:N, :]

        # model update
        model.train()
        if hy_params['opt_type']=='sgd':
            opt = torch.optim.SGD(model.parameters(), lr=hy_params['sgd_lr'],
                                  momentum=hy_params['sgd_momentum'],
                                  weight_decay=hy_params['sgd_weight_decay'])
        elif hy_params['opt_type']=='adam':
            opt = torch.optim.Adam(
                model.parameters(),
                lr=hy_params['sgd_lr'],
                weight_decay=hy_params.get('sgd_weight_decay', 0.0)
            )
        else:
            print("incorrect optim type")
            opt = None
        crossentropy_loss = nn.CrossEntropyLoss()
        Xb, Yb_oh, y_target = X_train[N-dN:N], Yc_iter_oh[N-dN:N], y_train[N-dN:N]
        Yb_oh = torch.as_tensor(Yb_oh, device=hy_params['device'], dtype=Xb.dtype)  # <-- key line
        y_target = torch.as_tensor(y_target, device=hy_params['device'], dtype=torch.long)  # <-- key line

        for epoch in range(hy_params['epochs']):
            for start in range(0, Xb.shape[0], hy_params['sgd_batch_size']):
                end = min(start + hy_params['sgd_batch_size'], Xb.shape[0])
                x_mb = Xb[start:end]
                y_mb = Yb_oh[start:end]

                logits = model(x_mb)
                if hy_params['criterion']=='mse':
                    loss = mse_loss(logits, y_mb) 
                elif hy_params['criterion']=='crossentropy':
                    y_mb_target = y_target[start:end]
                    loss = crossentropy_loss(logits, y_mb_target)
                else:
                    print("Using incorrect criterion type")
                    loss=None
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
        model.eval()
        tr_acc, tr_mse = accuracy_cnn(model, X_train[:N], y_train[:N], device = hy_params['device'])
        if hy_params['train_order_type'] == 'class_incremental':
            X_test_sub, y_test_sub = incre_seen_test_subset(y_train[:N], X_test, y_test)
            te_acc, te_mse = accuracy_cnn(model, X_test_sub, y_test_sub, device = hy_params['device'])
        elif hy_params['train_order_type'] == 'random':
            te_acc, te_mse = accuracy_cnn(model, X_test, y_test, device=hy_params['device'])
        elif hy_params['train_order_type'] == 'task_incremental':
            if _task_incre_use_full_test_eval(hy_params):
                te_acc, te_mse = accuracy_cnn(model, X_test, y_test, device=hy_params['device'])
            else:
                total_tasks = 5
                samples_per_task = len(Y_train_oh) // total_tasks
                current_task_idx = (N - 1) // samples_per_task
                X_test_sub, y_test_sub = incre_seen_test_subset_task_incremental(
                    X_test, y_test, current_task_idx, total_tasks
                )
                te_acc, te_mse = accuracy_cnn(model, X_test_sub, y_test_sub, device=hy_params['device'])
        else:
            te_acc, te_mse = None, None
            print("Train order type error")

        train_accs.append(tr_acc); test_accs.append(te_acc)
        train_mses.append(tr_mse); test_mses.append(te_mse)

    return np.array(train_accs), np.array(test_accs), np.array(train_mses), np.array(test_mses)

def sgd_train_streaming_vanilla_old(model, X_train, y_train, X_test, y_test, Niters, hy_params, gate_mode="none"):
    if hy_params['train_order_type'] == 'task_incremental':
        Y_train_oh = one_hot(y_train, 2)
    else:
        Y_train_oh = one_hot(y_train, 10)
    Yc_iter_oh = np.zeros((np.shape(Y_train_oh)))
    dN = Niters[1] - Niters[0]

    train_accs, test_accs = [], []
    train_mses, test_mses = [], []

    for Nidx, N in enumerate(Niters):
        if Nidx == 0:
            Yc_iter_oh[:N, :] = Y_train_oh[:N, :]
        else:
            Yc_iter_oh[N - dN:N, :] = Y_train_oh[N - dN:N, :]

        # model update
        model.train()
        if hy_params['opt_type']=='sgd':
            opt = torch.optim.SGD(model.parameters(), lr=hy_params['sgd_lr'],
                                  momentum=hy_params['sgd_momentum'],
                                  weight_decay=hy_params['sgd_weight_decay'])
        elif hy_params['opt_type']=='adam':
            opt = torch.optim.Adam(
                model.parameters(),
                lr=hy_params['sgd_lr'],
                weight_decay=hy_params.get('sgd_weight_decay', 0.0)
            )
        else:
            print("incorrect optim type")
            opt = None

        crossentropy_loss = nn.CrossEntropyLoss()
        Xb, Yb_oh, y_target = X_train[N-dN:N], Yc_iter_oh[N-dN:N], y_train[N-dN:N]
        Yb_oh = torch.as_tensor(Yb_oh, device=hy_params['device'], dtype=Xb.dtype)  # <-- key line
        y_target = torch.as_tensor(y_target, device=hy_params['device'], dtype=torch.long)  # <-- key line

        for epoch in range(hy_params['epochs']):
            for start in range(0, Xb.shape[0], hy_params['sgd_batch_size']):
                end = min(start + hy_params['sgd_batch_size'], Xb.shape[0])
                x_mb = Xb[start:end]
                y_mb = Yb_oh[start:end]

                logits = model(x_mb)
                if hy_params['criterion']=='mse':
                    loss = mse_loss(logits, y_mb)
                elif hy_params['criterion']=='crossentropy':
                    y_mb_target = y_target[start:end]
                    loss = crossentropy_loss(logits, y_mb_target)
                else:
                    print("Using incorrect criterion type")
                    loss=None
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
        model.eval()
        tr_acc, tr_mse = accuracy_cnn(model, X_train[:N], y_train[:N], device = hy_params['device'])
        if hy_params['train_order_type'] == 'class_incremental':
            X_test_sub, y_test_sub = incre_seen_test_subset(y_train[:N], X_test, y_test)
            te_acc, te_mse = accuracy_cnn(model, X_test_sub, y_test_sub, device = hy_params['device'])
        elif hy_params['train_order_type'] == 'random':
            te_acc, te_mse = accuracy_cnn(model, X_test, y_test, device=hy_params['device'])
        elif hy_params['train_order_type'] == 'task_incremental':
            if _task_incre_use_full_test_eval(hy_params):
                te_acc, te_mse = accuracy_cnn(model, X_test, y_test, device=hy_params['device'])
            else:
                total_tasks = 5
                samples_per_task = len(Y_train_oh) // total_tasks
                current_task_idx = (N - 1) // samples_per_task
                X_test_sub, y_test_sub = incre_seen_test_subset_task_incremental(
                    X_test, y_test, current_task_idx, total_tasks
                )
                te_acc, te_mse = accuracy_cnn(model, X_test_sub, y_test_sub, device=hy_params['device'])
        else:
            te_acc, te_mse = None, None
            print("Train order type error")

        train_accs.append(tr_acc); test_accs.append(te_acc)
        train_mses.append(tr_mse); test_mses.append(te_mse)


    return np.array(train_accs), np.array(test_accs), np.array(train_mses), np.array(test_mses)

def sgd_train_streaming_MSE_fixed_kernel_f0(model_vanilla, model, Kdata, X_train, y_train, X_test, y_test, Niters, hy_params, label_corrected=True):
    eta, gm, KbU_block_size = hy_params['eta'], hy_params['gm'], hy_params['KbU_block_size']

    if hy_params['train_order_type'] == 'task_incremental':
        Y_train_oh = one_hot(y_train, 2)
    else:
        Y_train_oh = one_hot(y_train, 10)
    Yc_iter_oh = np.zeros((np.shape(Y_train_oh)))
    dN = Niters[1] - Niters[0]
    IdN = np.eye(dN)

    train_accs, test_accs = [], []
    train_mses, test_mses = [], []

    model_vanilla.eval()
    for Nidx, N in enumerate(Niters):
        if Nidx == 0:
            Yc_iter_oh[:N, :] = Y_train_oh[:N, :]
        else:
            if label_corrected:
                X_next, Y_next = X_train[N - dN:N], Y_train_oh[N - dN:N]

                # logits: (dN, C)
                with torch.no_grad():
                    Y_logits_next = model_vanilla(X_next)  # torch, no grad

                ylog_np = Y_logits_next.detach().cpu().numpy()  # (dN, C), y_pred

                dyon_new = Y_train_oh[N - dN:N, :] - ylog_np  # (dN, 10)

                Moff_new = gm * IdN + Kdata[N - dN:N, N - dN:N]
                mini_bz = hy_params['sgd_batch_size']
                Mon_new = (1.0 / eta) * IdN + KbU_mini_batch(Kdata[N - dN:N, N - dN:N], mini_bz)
                Cnew = np.linalg.solve(Moff_new, Mon_new)

                Yc_iter_oh[N - dN:N, :] = np.dot(Cnew.T, dyon_new)
            else:
                Yc_iter_oh[N - dN:N, :] = Y_train_oh[N - dN:N, :]

        # model update
        model.train()
        device = torch.device(hy_params["device"])  # "cuda" or "cuda:0" or "cpu"
        model = model.to(device)
        opt = torch.optim.SGD(model.parameters(), lr=hy_params['sgd_lr'],
                              momentum=hy_params['sgd_momentum'],
                              weight_decay=hy_params['sgd_weight_decay'])
        Xb, Yb_oh = X_train[N-dN:N], Yc_iter_oh[N-dN:N]
        Xb = torch.as_tensor(X_train[N - dN:N], device=device, dtype=torch.float32)
        Yb_oh = torch.as_tensor(Yb_oh, device=hy_params['device'], dtype=Xb.dtype)  # <-- key line
        for start in range(0, Xb.shape[0], hy_params['sgd_batch_size']):
            end = min(start + hy_params['sgd_batch_size'], Xb.shape[0])
            x_mb = Xb[start:end]
            y_mb = Yb_oh[start:end]

            logits = model(x_mb)
            loss = mse_loss(logits, y_mb)  # your mse_loss expects same shape

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        model.eval()
        tr_acc, tr_mse = accuracy_cnn(model, X_train[:N], y_train[:N], device = hy_params['device'])
        if hy_params['train_order_type'] == 'class_incremental':
            X_test_sub, y_test_sub = incre_seen_test_subset(y_train[:N], X_test, y_test)
            te_acc, te_mse = accuracy_cnn(model, X_test_sub, y_test_sub, device = hy_params['device'])
        elif hy_params['train_order_type'] == 'random':
            te_acc, te_mse = accuracy_cnn(model, X_test, y_test, device=hy_params['device'])
        elif hy_params['train_order_type'] == 'task_incremental':
            if _task_incre_use_full_test_eval(hy_params):
                te_acc, te_mse = accuracy_cnn(model, X_test, y_test, device=hy_params['device'])
            else:
                total_tasks = 5
                samples_per_task = len(Y_train_oh) // total_tasks
                current_task_idx = (N - 1) // samples_per_task
                X_test_sub, y_test_sub = incre_seen_test_subset_task_incremental(
                    X_test, y_test, current_task_idx, total_tasks
                )
                te_acc, te_mse = accuracy_cnn(model, X_test_sub, y_test_sub, device=hy_params['device'])
        else:
            te_acc, te_mse = None, None
            print("Train order type error")
        train_accs.append(tr_acc); test_accs.append(te_acc)
        train_mses.append(tr_mse); test_mses.append(te_mse)

    return np.array(train_accs), np.array(test_accs), np.array(train_mses), np.array(test_mses)


def compute_Z_iter(K_train, Y_train_oh, sgd_on_gm, sgd_on_eta, on_cor_bs, online_cor_type, z_type, mini_bz):
    # z_type: 'both'/'online'/'offline'
    if online_cor_type == 'iter':
        Z_iter_oh = compute_corrected_Z_iter(K_train, Y_train_oh, gamma=sgd_on_gm, eta=sgd_on_eta, z_type=z_type, mini_bz=mini_bz)
    elif online_cor_type == 'full':
        Z_iter_oh = compute_corrected_Z_full(K_train, Y_train_oh, gamma=sgd_on_gm, eta=sgd_on_eta, mini_bz=mini_bz)
    elif online_cor_type == 'batch_gpu':
        Z_iter_oh = compute_corrected_Z_batch_gpu(K_train, Y_train_oh, gamma=sgd_on_gm, eta=sgd_on_eta,
                                                  batch_size=on_cor_bs, z_type=z_type, mini_bz=mini_bz)
    else:
        Z_iter_oh = None
        print("incorrect online correction type")
    return Z_iter_oh

import time

def sgd_train_streaming_MSE_on_cor_adap_kernel(model, Y_train_oh, X_train, y_train, X_test, y_test, Niters, hy_params,
                                              sgd_on_gm, sgd_on_eta, online_cor_type):
    adap_kernel_times, z_mini_bz = hy_params['adap_kernel_times'], hy_params['z_mini_bz']
    device, kernel_tp, z_type, adap_kernel_step = hy_params['device'], hy_params['kernel_tp'], hy_params['z_type'], int(len(X_train)/adap_kernel_times)

    model = model.to(device)
    opt = torch.optim.SGD(model.parameters(), lr=sgd_on_eta,
                          momentum=hy_params['sgd_momentum'],
                          weight_decay=hy_params['sgd_weight_decay'])
    dN = Niters[1] - Niters[0]
    if adap_kernel_step % dN != 0:
        raise ValueError(
            f"record_step ({dN}) must be a divisor of adap_kernel_step "
            f"(Ntrain/adap_kernel_times = {len(X_train)}/{adap_kernel_times} = {adap_kernel_step}). "
            f"Please set record_step to a value that divides {adap_kernel_step}."
        )

    train_accs, test_accs = [], []
    train_mses, test_mses = [], []
    K_train, K_tr_te = kernel_type(model, X_train[:adap_kernel_step], X_test, device, kernel_tp, hy_params)
    Z_iter_oh = compute_Z_iter(K_train, Y_train_oh[:adap_kernel_step], sgd_on_gm, sgd_on_eta, hy_params['on_cor_bs'], online_cor_type, z_type, mini_bz=z_mini_bz)

    for Nidx, N in enumerate(Niters):
        model.train()

        Xb = torch.as_tensor(X_train[N - dN:N], device=device, dtype=torch.float32)
        Yb_oh = torch.as_tensor(Z_iter_oh[N - dN:N], device=device, dtype=torch.float32)
        for start in range(0, Xb.shape[0], hy_params['sgd_batch_size']):
            end = min(start + hy_params['sgd_batch_size'], Xb.shape[0])
            x_mb = Xb[start:end]
            y_mb = Yb_oh[start:end]

            logits = model(x_mb)

            loss = mse_loss(logits, y_mb)
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"FATAL: NaN loss detected at step {N} . Terminating run.")
                break
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        model.eval()
        tr_acc, tr_mse = accuracy_cnn(model, X_train[:N], y_train[:N], device=hy_params['device'])
        if hy_params['train_order_type'] == 'class_incremental':
            X_test_sub, y_test_sub = incre_seen_test_subset(y_train[:N], X_test, y_test)
            te_acc, te_mse = accuracy_cnn(model, X_test_sub, y_test_sub, device=hy_params['device'])
        elif hy_params['train_order_type'] == 'random':
            te_acc, te_mse = accuracy_cnn(model, X_test, y_test, device=hy_params['device'])
        elif hy_params['train_order_type'] == 'task_incremental':
            if _task_incre_use_full_test_eval(hy_params):
                te_acc, te_mse = accuracy_cnn(model, X_test, y_test, device=hy_params['device'])
            else:
                total_tasks = 5
                samples_per_task = len(Y_train_oh) // total_tasks
                current_task_idx = (N - 1) // samples_per_task
                X_test_sub, y_test_sub = incre_seen_test_subset_task_incremental(
                    X_test, y_test, current_task_idx, total_tasks
                )
                te_acc, te_mse = accuracy_cnn(model, X_test_sub, y_test_sub, device=hy_params['device'])
        else:
            te_acc, te_mse = None, None
            print("Train order type error")
        train_accs.append(tr_acc);
        test_accs.append(te_acc)
        train_mses.append(tr_mse);
        test_mses.append(te_mse)

        # update kernel
        model.eval()
        if N % adap_kernel_step == 0 and Nidx < len(Niters) - 1:
            print(f"step={N} when updating the kernel")
            # update of the Z_iter_oh and Kernel
            K_train, K_tr_te = kernel_type(model, X_train[:N+adap_kernel_step], X_test, device, kernel_tp, hy_params)
            Z_iter_oh = compute_Z_iter(K_train, Y_train_oh[:N+adap_kernel_step], sgd_on_gm, sgd_on_eta,
                                       hy_params['on_cor_bs'], online_cor_type, z_type, mini_bz=z_mini_bz)


    return np.array(train_accs), np.array(test_accs), np.array(train_mses), np.array(test_mses)

def sgd_train_streaming_MSE_online_correction_epochs(model, Y_train_oh, X_train, y_train, X_test, y_test, Niters, hy_params, sgd_on_gm, sgd_on_eta, online_cor_type, fixed_first_ep_kernel):
    device = torch.device(hy_params["device"])
    kernel_tp = hy_params['kernel_tp']
    z_mini_bz = hy_params['z_mini_bz']

    model = model.to(device)
    if hy_params['sgd_momentum'] > 0:
        set_nesterov = True
    else:
        set_nesterov = False
    opt = torch.optim.SGD(model.parameters(), lr=hy_params['sgd_lr'],
                          momentum=hy_params['sgd_momentum'],
                          weight_decay=hy_params['sgd_weight_decay'], nesterov=set_nesterov)
    dN = Niters[1] - Niters[0]

    train_accs, test_accs = [], []
    train_mses, test_mses = [], []
    train_accs_epochs, test_accs_epochs = [], []

    for epoch in range(hy_params['epochs']):
        print(f"epoch number is {epoch}")
        Begin_K = time.time()
        K_train, K_tr_te = kernel_type(model, X_train, X_test, device, kernel_tp, hy_params)
        End_K = time.time()
        print("time cost in calculating Kernel matrix per epoch:", End_K-Begin_K)

        print(f"First 5x5 elements of the train {kernel_tp} Kernel Matrix in the epoch {epoch}:")
        print(K_train[:5, :5])

        if fixed_first_ep_kernel:
            if epoch==0:
                Begin_Z = time.time()
                Z_iter_oh = compute_Z_iter(K_train, Y_train_oh, sgd_on_gm, sgd_on_eta, hy_params['on_cor_bs'], online_cor_type, hy_params['z_type'], mini_bz=z_mini_bz)
                print(f"First 50-55 elements of the corrected labels using {online_cor_type} in epoch {epoch}:")
                print(Z_iter_oh[50:55])
                End_Z = time.time()
                print("full time cost in calculating Z_iter_oh:", End_Z-Begin_Z)
        else:
            Begin_Z = time.time()
            Z_iter_oh = compute_Z_iter(K_train, Y_train_oh, sgd_on_gm, sgd_on_eta, hy_params['on_cor_bs'],
                                       online_cor_type, hy_params['z_type'], mini_bz=z_mini_bz)
            print(f"First 50-55 elements of the corrected labels using {online_cor_type} in epoch {epoch}:")
            print(Z_iter_oh[50:55])
            End_Z = time.time()
            print("full time cost in calculating Z_iter_oh:", End_Z - Begin_Z)

        print("##########")
        for Nidx, N in enumerate(Niters):
            # model update
            model.train()

            Xb = torch.as_tensor(X_train[N - dN:N], device=device, dtype=torch.float32)
            Yb_oh = torch.as_tensor(Z_iter_oh[N - dN:N], device=device, dtype=torch.float32)
            for start in range(0, Xb.shape[0], hy_params['sgd_batch_size']):
                end = min(start + hy_params['sgd_batch_size'], Xb.shape[0])
                x_mb = Xb[start:end]
                y_mb = Yb_oh[start:end]

                logits = model(x_mb)

                loss = mse_loss(logits, y_mb)  # your mse_loss expects same shape
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"FATAL: NaN loss detected at step {N}. Terminating run.")
                    break

                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
            model.eval()
            tr_acc, tr_mse = accuracy_cnn(model, X_train[:N], y_train[:N], device = hy_params['device'])
            if hy_params['train_order_type'] == 'class_incremental':
                X_test_sub, y_test_sub = incre_seen_test_subset(y_train[:N], X_test, y_test)
                te_acc, te_mse = accuracy_cnn(model, X_test_sub, y_test_sub, device = hy_params['device'])
            elif hy_params['train_order_type'] == 'random':
                te_acc, te_mse = accuracy_cnn(model, X_test, y_test, device=hy_params['device'])
            elif hy_params['train_order_type'] == 'task_incremental':
                if _task_incre_use_full_test_eval(hy_params):
                    te_acc, te_mse = accuracy_cnn(model, X_test, y_test, device=hy_params['device'])
                else:
                    total_tasks = 5
                    samples_per_task = len(Y_train_oh) // total_tasks
                    current_task_idx = (N - 1) // samples_per_task
                    X_test_sub, y_test_sub = incre_seen_test_subset_task_incremental(
                        X_test, y_test, current_task_idx, total_tasks
                    )
                    te_acc, te_mse = accuracy_cnn(model, X_test_sub, y_test_sub, device=hy_params['device'])
            else:
                te_acc, te_mse = None, None
                print("Train order type error")
            train_accs.append(tr_acc); test_accs.append(te_acc)
            train_mses.append(tr_mse); test_mses.append(te_mse)
        train_accs_epochs.append(train_accs[-1]); test_accs_epochs.append(test_accs[-1])

    return train_accs_epochs, test_accs_epochs

def expand_kernel_multi_epoch(K, num_tasks, num_epochs):
    is_numpy = isinstance(K, np.ndarray)
    K = torch.as_tensor(K)

    N_row, N_col = K.shape

    # Calculate samples per task for rows and cols
    S_row = N_row // num_tasks
    S_col = N_col // num_tasks

    # 1. Reshape into (Tasks, Samples_Row, Tasks, Samples_Col)
    K_reshaped = K.view(num_tasks, S_row, num_tasks, S_col)

    # 2. Add the Epoch dimensions: (Tasks, 1, Samples_Row, Tasks, 1, Samples_Col)
    K_expanded = K_reshaped.unsqueeze(1).unsqueeze(4)

    # 3. Duplicate along the Epoch dimensions
    K_multi = K_expanded.expand(num_tasks, num_epochs, S_row,
                                num_tasks, num_epochs, S_col)

    # 4. Flatten back into a 2D matrix
    K_final = K_multi.reshape(num_tasks * num_epochs * S_row,
                              num_tasks * num_epochs * S_col).contiguous()

    if is_numpy:
        return K_final.numpy()

    return K_final


def expand_data_multi_epoch(data, num_tasks, num_epochs=3):
    # etect if input is numpy, and convert to tensor for reshaping
    is_numpy = isinstance(data, np.ndarray)
    tensor = torch.as_tensor(data)

    # Number of samples per single task block
    S = len(tensor) // num_tasks

    # Shape: (Tasks, Samples, *rest_of_dimensions)
    reshaped = tensor.view(num_tasks, S, *tensor.shape[1:])

    # Shape: (Tasks, 1, Samples, *rest_of_dimensions)
    expanded = reshaped.unsqueeze(1)

    # Shape: (Tasks, Epochs, Samples, *rest_of_dimensions)
    repeated = expanded.expand(num_tasks, num_epochs, S, *tensor.shape[1:])

    # Flatten the Tasks and Epochs into a single sequential dimension
    final_tensor = repeated.reshape(-1, *tensor.shape[1:]).contiguous()

    # Convert back to numpy if the original was numpy
    if is_numpy:
        return final_tensor.numpy()

    return final_tensor


def optimizer(hy_params, model):
    if hy_params['opt_type'] == 'sgd':
        opt = torch.optim.SGD(
            model.parameters(),
            lr=hy_params['sgd_lr'],
            momentum=hy_params['sgd_momentum'],
            weight_decay=hy_params['sgd_weight_decay']
        )
    elif hy_params['opt_type'] == 'adam':
        opt = torch.optim.Adam(
            model.parameters(),
            lr=hy_params['sgd_lr'],
            weight_decay=hy_params.get('sgd_weight_decay', 0.0)
        )
    else:
        opt = None
        raise ValueError("incorrect optim type")
    return opt

def verify_multi_epoch_expansion(total_tasks, num_epochs,
                                 y_train_base, y_test_base, y_train_multi,
                                 K_train_base, K_train_multi,
                                 K_tr_te_base, K_tr_te_multi):
    # The length of a single task's data
    S_train_task = len(y_train_base) // total_tasks
    S_test_task = len(y_test_base) // total_tasks

    print("\n" + "="*55)
    print("1. LABEL VERIFICATION (Proving 1-1-1-2-2-2 pattern)")
    print("="*55)

    print(f"Original y_train Task 1 (First 10): {y_train_base[:10].tolist()}")
    print(f"Expanded y_train Task 1 Epoch 1:    {y_train_multi[:10].tolist()}")

    # Jump exactly one TASK length forward to see Epoch 2 of Task 1
    print(f"Expanded y_train Task 1 Epoch 2:    {y_train_multi[S_train_task : S_train_task + 10].tolist()}")

    # Jump forward by (Epochs * Task Length) to see the start of Task 2
    start_task_2 = num_epochs * S_train_task
    print(f"Expanded y_train Task 2 Epoch 1:    {y_train_multi[start_task_2 : start_task_2 + 10].tolist()}")

    print("\n" + "="*55)
    print("2. KERNEL MATRIX VERIFICATION (Checking top-left blocks)")
    print("="*55)
    print("--- Original K_train (Task 1: First 4x4) ---")
    print(K_train_base[:4, :4])

    print("\n--- Expanded K_train (Task 1, Epoch 1: First 4x4) ---")
    print(K_train_multi[:4, :4])

    print("\n--- Expanded K_train (Task 1, Epoch 2: First 4x4) ---")
    # Jump exactly one task length in both rows and columns to find the Epoch 2 x Epoch 2 block
    print(K_train_multi[S_train_task : S_train_task + 4,
                        S_train_task : S_train_task + 4])

    print("\n" + "="*55)
    print("3. K_tr_te CROSS-KERNEL VERIFICATION")
    print("="*55)
    print("--- Original K_tr_te (Task 1: First 4x4) ---")
    print(K_tr_te_base[:4, :4])

    print("\n--- Expanded K_tr_te (Task 1: Train Epoch 2 vs Test Epoch 1) ---")
    # Jump 1 task length down the rows (Train Epoch 2), but stay at the start of columns (Test Epoch 1)
    print(K_tr_te_multi[S_train_task : S_train_task + 4, 0 : 4])
    print("="*55 + "\n")


def sgd_train_task_incre_label_correction_epochs_adap_kernel(model, X_train, y_train, X_test, y_test, hy_params, gate_mode='none', num_epochs=3, sgd_on_gm=100, sgd_on_eta=0.01, online_cor_type='batch_gpu'): # gate_mode="none", "infer"
    adap_kernel_times = hy_params['adap_kernel_times']
    total_tasks = getattr(model, "num_tasks", 5)

    device, kernel_tp, z_type, adap_kernel_step = hy_params['device'], hy_params['kernel_tp'], hy_params['z_type'], int(len(X_train)*num_epochs/adap_kernel_times)
    adap_kernel_epoch, z_mini_bz = int(num_epochs*total_tasks/adap_kernel_times), hy_params['z_mini_bz']
    model = model.to(device)

    # toggle gating on/off at model level
    if hasattr(model, "set_gating"):
        model.set_gating(gate_mode != "none")
    else:
        # fallback
        if hasattr(model, "use_gating"):
            model.use_gating = (gate_mode != "none")


    # Expand the datasets to multi-epoch (Task1-Task1-Task1...)
    X_train_multi = expand_data_multi_epoch(X_train, total_tasks, num_epochs)
    y_train_multi = expand_data_multi_epoch(y_train, total_tasks, num_epochs)
    Y_train_oh_multi = one_hot(y_train_multi, 2)

    K_train_multi, K_tr_te_multi = kernel_type(model, X_train_multi[:adap_kernel_step], X_test, device, kernel_tp, hy_params)
    Z_iter_oh_multi = compute_Z_iter(K_train_multi, Y_train_oh_multi[:adap_kernel_step], sgd_on_gm, sgd_on_eta, hy_params['on_cor_bs'], online_cor_type, z_type, mini_bz=z_mini_bz)

    opt = torch.optim.SGD(model.parameters(), lr=sgd_on_eta,
                          momentum=hy_params['sgd_momentum'],
                          weight_decay=hy_params['sgd_weight_decay'])

    total_epochs, epoch_steps = num_epochs*total_tasks, []
    train_accs, test_accs, train_mses, test_mses = [], [], [], []

    tr_samples_per_task,  te_samples_per_task = len(X_train) // total_tasks, len(X_test) // total_tasks
    tr_samples_per_task_multi_epoch, te_samples_per_task_multi_epoch = tr_samples_per_task*num_epochs, te_samples_per_task*num_epochs

    for epoch in range(total_epochs):
        epoch_steps.append(epoch+1)
        task_id, epoch_id = int(epoch/num_epochs), int(epoch%num_epochs)

        print("current task:", task_id)
        # ---- optimizer (your original structure) ----
        model.train()

        # ---- get current training window (your original logic) ----
        task_start, task_end = task_id*tr_samples_per_task_multi_epoch+epoch_id*tr_samples_per_task, task_id*tr_samples_per_task_multi_epoch+(epoch_id+1)*tr_samples_per_task
        Xb, Zb_oh, y_target = X_train_multi[task_start:task_end], Z_iter_oh_multi[task_start:task_end], y_train_multi[task_start:task_end]
        Xb = torch.as_tensor(Xb, device=device, dtype=torch.float32)
        Zb_oh = torch.as_tensor(Zb_oh, device=hy_params['device'], dtype=Xb.dtype)

        # ---- SGD steps ----
        for start in range(0, Xb.shape[0], hy_params['sgd_batch_size']):
            end = min(start + hy_params['sgd_batch_size'], Xb.shape[0])
            x_mb = Xb[start:end]
            z_mb = Zb_oh[start:end]

            # Baseline model (no gating)
            logits = model(x_mb)
            # 1a) classification loss
            loss_cls = mse_loss(logits, z_mb)

            # --- NEW: NaN/Inf Safety Check ---
            if torch.isnan(loss_cls) or torch.isinf(loss_cls):
                print(f"FATAL: NaN loss detected at epoch {epoch}, batch_start {start}. Terminating current run.")
                break
            opt.zero_grad(set_to_none=True)
            loss_cls.backward()
            opt.step()

        # ---- evaluation ----
        model.eval()
        tr_eval_end = (task_id+1)*tr_samples_per_task
        tr_acc, tr_mse = accuracy_cnn(model, X_train[:tr_eval_end], y_train[:tr_eval_end], device=hy_params['device'], gate_mode="none", true_gate_ids=None)

        te_gate_mode = "none"
        if _task_incre_use_full_test_eval(hy_params):
            X_te, y_te = X_test, y_test
        else:
            X_te, y_te = incre_seen_test_subset_task_incremental(
                X_test, y_test, task_id, total_tasks
            )

        # Generate an array mapping the test set to its specific tasks (ordered by task blocks)
        te_task_ids = torch.arange(len(y_te)) // te_samples_per_task
        te_acc, te_mse = accuracy_cnn(
            model, X_te, y_te,
            device=hy_params['device'],
            gate_mode=te_gate_mode,
            true_gate_ids=te_task_ids
        )

        train_accs.append(tr_acc); test_accs.append(te_acc)
        train_mses.append(tr_mse); test_mses.append(te_mse)

        # update kernel
        model.eval()
        if int(epoch+1) % adap_kernel_epoch == 0 and int(epoch+1) < total_epochs:
            print(f"epoch={epoch} when updating the kernel")
            next_update_end = int(((epoch+1)/num_epochs + 1)*adap_kernel_step)
            # update of the Z_iter_oh and Kernel
            K_train_multi, K_tr_te_multi = kernel_type(model, X_train_multi[:next_update_end], X_test, device,
                                                       kernel_tp, hy_params)
            Z_iter_oh_multi = compute_Z_iter(K_train_multi, Y_train_oh_multi[:next_update_end], sgd_on_gm, sgd_on_eta,
                                             hy_params['on_cor_bs'], online_cor_type, z_type, mini_bz=z_mini_bz)
        else:
            print("there's no kernel updating")

    return epoch_steps, np.array(train_accs), np.array(test_accs), np.array(train_mses), np.array(test_mses)


def ewc_penalty_multi_tasks(model, fishers, theta_stars):
    loss = torch.zeros((), device=next(model.parameters()).device)

    # Iterate over every previous task 'k'
    for fisher_k, theta_star_k in zip(fishers, theta_stars):
        for n, p in model.named_parameters():
            if (not p.requires_grad) or (n not in fisher_k):
                continue

            # Calculate penalty against task k's specific anchor and Fisher
            loss = loss + (fisher_k[n] * (p - theta_star_k[n]).pow(2)).sum()

    return 0.5 * loss

def sgd_train_streaming_gating_ewc(model, X_train, y_train, X_test, y_test, Niters, hy_params, gate_mode="none", *, task_boundary_replay=False):
    device = torch.device(hy_params["device"])
    model = model.to(device)
    total_tasks = getattr(model, "num_tasks", 5)
    num_train_per_task = int(len(X_train)/total_tasks)
    print("num_train per task:", num_train_per_task)

    # toggle gating on/off at model level
    if hasattr(model, "set_gating"):
        model.set_gating(gate_mode != "none")
    else:
        # fallback
        if hasattr(model, "use_gating"):
            model.use_gating = (gate_mode != "none")

    if hy_params['train_order_type'] == 'task_incremental':
        Y_train_oh = one_hot(y_train, 2)
    else:
        Y_train_oh = one_hot(y_train, 10)
    dN = Niters[1] - Niters[0]

    train_accs, test_accs = [], []
    train_mses, test_mses = [], []
    if hy_params['opt_type'] == 'sgd':
        opt = torch.optim.SGD(
            model.parameters(),
            lr=hy_params['sgd_lr'],
            momentum=hy_params['sgd_momentum'],
            weight_decay=hy_params['sgd_weight_decay']
        )
    elif hy_params['opt_type'] == 'adam':
        opt = torch.optim.Adam(
            model.parameters(),
            lr=hy_params['sgd_lr'],
            weight_decay=hy_params.get('sgd_weight_decay', 0.0)
        )
    else:
        raise ValueError("incorrect optim type")
    crossentropy_loss = nn.CrossEntropyLoss()
    device = torch.device(hy_params["device"])
    model = model.to(device)

    # --- EWC state ---
    # Initialize empty lists to store history for Eq 1
    fishers = []
    theta_stars = []

    def _cumulative_replay_prefix(n_cum: int, step_n: int) -> None:
        """One or more shuffled passes over X_train[:n_cum] with current opt / EWC / gate rules."""
        replay_epochs = int(hy_params.get('replay_epochs', 1))
        if n_cum <= 0:
            return
        X_np = X_train[:n_cum]
        y_np = y_train[:n_cum]
        Y_oh_np = Y_train_oh[:n_cum]
        samples_pt = num_train_per_task
        for _ in range(replay_epochs):
            perm = np.random.permutation(n_cum)
            for rs in range(0, n_cum, hy_params['sgd_batch_size']):
                re = min(rs + hy_params['sgd_batch_size'], n_cum)
                idx_batch = perm[rs:re]
                x_mb = torch.as_tensor(X_np[idx_batch], device=device, dtype=torch.float32)
                y_mb_target = torch.as_tensor(y_np[idx_batch], device=device, dtype=torch.long)
                y_mb = torch.as_tensor(Y_oh_np[idx_batch], device=device, dtype=x_mb.dtype)

                if gate_mode != "none":
                    if hasattr(model, "num_tasks"):
                        task_ids = (idx_batch // samples_pt).astype(np.int64)
                        task_ids = np.minimum(task_ids, model.num_tasks - 1)
                        gate_ids_batch = torch.as_tensor(task_ids, device=device, dtype=torch.long)
                        logits = model(x_mb, gate_ids=gate_ids_batch)
                    else:
                        logits = model(x_mb, gate_ids=y_mb_target)
                else:
                    logits = model(x_mb)

                if hy_params['criterion'] == 'mse':
                    loss_cls = mse_loss(logits, y_mb)
                elif hy_params['criterion'] == 'crossentropy':
                    loss_cls = crossentropy_loss(logits, y_mb_target)
                else:
                    raise ValueError("incorrect criterion type")

                if len(fishers) > 0 and len(theta_stars) > 0:
                    loss_ewc = ewc_penalty_multi_tasks(model, fishers, theta_stars)
                    loss = loss_cls + hy_params["ewc_lambda"] * loss_ewc
                else:
                    loss = loss_cls

                if torch.isnan(loss_cls) or torch.isinf(loss_cls):
                    print(f"FATAL: NaN loss during replay at step {step_n}. Terminating run.")
                    break
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()

    for Nidx, N in enumerate(Niters):
        # ---- optimizer (your original structure) ----
        model.train()

        # ---- get current training window (your original logic) ----
        Xb, Yb_oh, y_target = X_train[N - dN:N], Y_train_oh[N - dN:N], y_train[N - dN:N]
        Xb = torch.as_tensor(X_train[N - dN:N], device=device, dtype=torch.float32)
        Yb_oh = torch.as_tensor(Yb_oh, device=hy_params['device'], dtype=Xb.dtype)
        y_target = torch.as_tensor(y_target, device=hy_params['device'], dtype=torch.long)

        # ---- SGD steps ----
        for start in range(0, Xb.shape[0], hy_params['sgd_batch_size']):
            end = min(start + hy_params['sgd_batch_size'], Xb.shape[0])
            x_mb = Xb[start:end]
            y_mb = Yb_oh[start:end]  # one-hot (float) for MSE case
            y_mb_target = y_target[start:end]  # LongTensor [B]

            if gate_mode != "none":
                if hasattr(model, "num_tasks"):
                    # --- Task-Incremental Gating ---
                    # 1. Determine how many samples are in a single task block
                    samples_per_task = len(y_train) // model.num_tasks

                    # 2. Calculate the current Task ID (0 to num_tasks - 1) based on step N
                    current_task_id = (N - 1) // samples_per_task

                    # Safety clamp (just in case N slightly exceeds total due to batching math)
                    current_task_id = min(current_task_id, model.num_tasks - 1)

                    # 3. Create a gate tensor matching the batch size, filled with the Task ID
                    gate_ids_batch = torch.full_like(y_mb_target, current_task_id)

                    logits = model(x_mb, gate_ids=gate_ids_batch)
                else:
                    # --- Class-Incremental Gating ---
                    # Fall back to using the target label as the gate ID
                    logits = model(x_mb, gate_ids=y_mb_target)
            else:
                # Baseline model (no gating)
                logits = model(x_mb)
            # 1a) classification loss
            if hy_params['criterion'] == 'mse':
                loss_cls = mse_loss(logits, y_mb)
            elif hy_params['criterion'] == 'crossentropy':
                loss_cls = crossentropy_loss(logits, y_mb_target)
            else:
                raise ValueError("incorrect criterion type")

            # --- add EWC penalty if available ---
            if len(fishers) > 0 and len(theta_stars) > 0:
                # Pass the LISTS to the new penalty function
                loss_ewc = ewc_penalty_multi_tasks(model, fishers, theta_stars)
                loss = loss_cls + hy_params["ewc_lambda"] * loss_ewc

                # Check if N is exactly at the start of a new task boundary
                is_new_task_step = ((N - dN) % num_train_per_task == 0)
                # 'start == 0' ensures we only print on the very first mini-batch of that step
                if is_new_task_step and start==0:
                    print(f"\n--- New Task Started (N={N-dN}) ---")
                    # Use .item() to extract the clean float value from the PyTorch tensor
                    print(f"Classification Loss: {loss_cls.item():.6f}")
                    print(f"EWC Loss (\u03BB=1):      {loss_ewc.item():.6f}")
                    print(f"Total Loss:          {loss.item():.6f}\n")
            else:
                loss = loss_cls

            if torch.isnan(loss_cls) or torch.isinf(loss_cls):
                print(f"FATAL: NaN loss detected at step {N}. Terminating run.")
                break
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

        # ---- optional cumulative replay (upper-bound-style baseline) ----
        if task_boundary_replay and hy_params['train_order_type'] == 'task_incremental' and N > 0:
            sched = hy_params.get('replay_schedule', 'task_boundary')
            if sched == 'task_boundary':
                do_replay = (N % num_train_per_task) == 0
            elif sched == 'every_chunk':
                do_replay = True
            else:
                raise ValueError(
                    f"hy_params['replay_schedule'] must be 'task_boundary' or 'every_chunk', got {sched!r}"
                )
            if do_replay:
                _cumulative_replay_prefix(N, N)

        # ---- after training this chunk: update EWC state (theta_star, fisher) ----
        if (N % num_train_per_task) == 0 and hy_params['ewc_lambda'] > 0:
            # 1. Get the optimal weights for the task we just finished
            theta_star_new = snapshot_params(model)

            # 2. Estimate the Fisher matrix for the task we just finished
            fisher_new = estimate_fisher_diag(
                model,
                X_train[N - num_train_per_task:N], y_train[N - num_train_per_task:N],
                device=device,
                criterion=hy_params["criterion"],
                fisher_n_samples=hy_params.get("ewc_fisher_n", None),
            )

            # 3. Append them to our history lists (Equation 1 behavior)
            theta_stars.append(theta_star_new)
            fishers.append(fisher_new)

        # ---- evaluation ----
        model.eval()

        # 1. We need to know the block sizes to generate accurate Task IDs
        tr_samples_per_task = len(Y_train_oh) // total_tasks
        te_samples_per_task = len(X_test) // total_tasks
        # ==========================================
        # Train metric: oracle gating makes most sense if gating is enabled
        # ==========================================
        tr_gate_mode = gate_mode if getattr(model, "use_gating", False) else "none"
        # Generate an array of [0,0,0..., 1,1,1..., etc] matching the current length N
        tr_task_ids = torch.arange(N) // tr_samples_per_task

        tr_acc, tr_mse = accuracy_cnn(
            model, X_train[:N], y_train[:N],
            device=hy_params['device'],
            gate_mode=tr_gate_mode,
            true_gate_ids=tr_task_ids if hy_params['train_order_type'] == 'task_incremental' else None
        )
        # Test metric: depends on gate_mode
        te_gate_mode = gate_mode if getattr(model, "use_gating", False) else "none"
        if hy_params['train_order_type'] == 'class_incremental':
            X_test_sub, y_test_sub = incre_seen_test_subset(y_train[:N], X_test, y_test)
            te_acc, te_mse = accuracy_cnn(
                model, X_test_sub, y_test_sub,
                device=hy_params['device'],
                gate_mode=te_gate_mode)  # "infer" uses Option A; "oracle" uses true labels; "none" baseline
        elif hy_params['train_order_type'] == 'random':
            te_acc, te_mse = accuracy_cnn(
                model, X_test, y_test,
                device=hy_params['device'],
                gate_mode=te_gate_mode)
        elif hy_params['train_order_type'] == 'task_incremental':
            # Full test set; per-sample task id for oracle / infer gating (test ordered by task blocks).
            te_task_ids = torch.arange(len(X_test)) // te_samples_per_task
            te_acc, te_mse = accuracy_cnn(
                model, X_test, y_test,
                device=hy_params['device'],
                gate_mode=te_gate_mode,
                true_gate_ids=te_task_ids,
            )
        else:
            raise ValueError("Train order type error")

        train_accs.append(tr_acc); test_accs.append(te_acc)
        train_mses.append(tr_mse); test_mses.append(te_mse)

    return np.array(train_accs), np.array(test_accs), np.array(train_mses), np.array(test_mses)


def sgd_train_streaming_gating_ewc_cumulative(model, X_train, y_train, X_test, y_test, Niters, hy_params, gate_mode="none"):
    return sgd_train_streaming_gating_ewc(
        model, X_train, y_train, X_test, y_test, Niters, hy_params, gate_mode,
        task_boundary_replay=True,
    )


def sgd_train_streaming_task_incre_epochs(model, X_train, y_train, X_test, y_test, hy_params, gate_mode='none', num_epochs=3): # gate_mode="none", "infer"

    device = torch.device(hy_params["device"])
    model = model.to(device)

    # toggle gating on/off at model level
    if hasattr(model, "set_gating"):
        model.set_gating(gate_mode != "none")
    else:
        # fallback
        if hasattr(model, "use_gating"):
            model.use_gating = (gate_mode != "none")

    opt = optimizer(hy_params, model)
    crossentropy_loss = nn.CrossEntropyLoss()
    device = torch.device(hy_params["device"])
    model = model.to(device)

    # --- EWC state ---
    fishers = []
    theta_stars = []
    train_accs, test_accs = [], []
    train_mses, test_mses = [], []
    total_tasks = getattr(model, "num_tasks", 5)
    tr_samples_per_task = len(X_train) // total_tasks
    te_samples_per_task = len(X_test) // total_tasks
    Y_train_oh, total_epochs, epoch_steps = one_hot(y_train, 2), num_epochs*total_tasks, []

    for epoch in range(total_epochs):
        epoch_steps.append(epoch+1)
        task_id = int(epoch/num_epochs)
        print("current task:", task_id)
        # ---- optimizer (your original structure) ----
        model.train()

        # ---- get current training window (your original logic) ----
        task_start, task_end = task_id*tr_samples_per_task, (task_id+1)*tr_samples_per_task
        Xb, Yb_oh, y_target = X_train[task_start:task_end], Y_train_oh[task_start:task_end], y_train[task_start:task_end]
        Xb = torch.as_tensor(X_train[task_start:task_end], device=device, dtype=torch.float32)
        Yb_oh = torch.as_tensor(Yb_oh, device=hy_params['device'], dtype=Xb.dtype)
        y_target = torch.as_tensor(y_target, device=hy_params['device'], dtype=torch.long)

        # ---- SGD steps ----
        for start in range(0, Xb.shape[0], hy_params['sgd_batch_size']):
            end = min(start + hy_params['sgd_batch_size'], Xb.shape[0])
            x_mb = Xb[start:end]
            y_mb = Yb_oh[start:end]  # one-hot (float) for MSE case
            y_mb_target = y_target[start:end]  # LongTensor [B]

            # Baseline model (no gating)
            logits = model(x_mb)
            # 1a) classification loss
            if hy_params['criterion'] == 'mse':
                loss_cls = mse_loss(logits, y_mb)
            elif hy_params['criterion'] == 'crossentropy':
                loss_cls = crossentropy_loss(logits, y_mb_target)
            else:
                raise ValueError("incorrect criterion type")

            # --- add EWC penalty if available ---
            if len(fishers) > 0 and len(theta_stars) > 0:
                loss = loss_cls + hy_params["ewc_lambda"] * ewc_penalty_multi_tasks(model, fishers, theta_stars)
            else:
                loss = loss_cls
            if torch.isnan(loss_cls) or torch.isinf(loss_cls):
                print(f"FATAL: NaN loss detected. Terminating run.")
                break
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

        # ---- after training this chunk: update EWC state (theta_star, fisher) ----
        if (epoch+1) % num_epochs == 0 and hy_params['ewc_lambda'] > 0:
            print(f"update fisher in epoch {epoch}")
            # 1. Get the optimal weights for the task we just finished
            theta_star_new = snapshot_params(model)

            # 2. Estimate the Fisher matrix for the task we just finished
            fisher_new = estimate_fisher_diag(
                model,
                Xb, y_target,
                device=device,
                criterion=hy_params["criterion"],
                fisher_n_samples=hy_params.get("ewc_fisher_n", None),
            )
            theta_stars.append(theta_star_new)
            fishers.append(fisher_new)

        model.eval()

        tr_acc, tr_mse = accuracy_cnn(
            model, X_train[:task_end], y_train[:task_end],
            device=hy_params['device'],
            gate_mode="none",
            true_gate_ids=None
        )
        # Test metric: depends on gate_mode
        te_gate_mode = "none"
        if _task_incre_use_full_test_eval(hy_params):
            X_te, y_te = X_test, y_test
        else:
            X_te, y_te = incre_seen_test_subset_task_incremental(
                X_test, y_test, task_id, total_tasks
            )

        # Generate an array mapping the test set to its specific tasks (ordered by task blocks)
        te_task_ids = torch.arange(len(y_te)) // te_samples_per_task
        te_acc, te_mse = accuracy_cnn(
            model, X_te, y_te,
            device=hy_params['device'],
            gate_mode=te_gate_mode,
            true_gate_ids=te_task_ids
        )

        train_accs.append(tr_acc); test_accs.append(te_acc)
        train_mses.append(tr_mse); test_mses.append(te_mse)

    return epoch_steps, np.array(train_accs), np.array(test_accs), np.array(train_mses), np.array(test_mses)


def sgd_train_streaming_gating_ewc_epochs(model, X_train, y_train, X_test, y_test, Niters, hy_params, gate_mode="none", ewc_lambda=0): # gate_mode="none", "infer"
    hy_params.update({'ewc_lambda': ewc_lambda})
    device = torch.device(hy_params["device"])
    model = model.to(device)

    # toggle gating on/off at model level
    if hasattr(model, "set_gating"):
        model.set_gating(gate_mode != "none")
    else:
        # fallback
        if hasattr(model, "use_gating"):
            model.use_gating = (gate_mode != "none")

    if hy_params['train_order_type'] == 'task_incremental':
        Y_train_oh = one_hot(y_train, 2)
    else:
        Y_train_oh = one_hot(y_train, 10)

    dN = Niters[1] - Niters[0]

    train_accs, test_accs = [], []
    train_mses, test_mses = [], []
    train_accs_epochs, test_accs_epochs = [], []
    if hy_params['sgd_momentum'] > 0:
        set_nesterov = True
    else:
        set_nesterov = False
    if hy_params['opt_type'] == 'sgd':
        opt = torch.optim.SGD(model.parameters(), lr=hy_params['sgd_lr'],
                              momentum=hy_params['sgd_momentum'],
                              weight_decay=hy_params['sgd_weight_decay'], nesterov=set_nesterov)
    elif hy_params['opt_type'] == 'adam':
        opt = torch.optim.Adam(
            model.parameters(),
            lr=hy_params['sgd_lr'],
            weight_decay=hy_params.get('sgd_weight_decay', 0.0)
        )
    else:
        raise ValueError("incorrect optim type")
    crossentropy_loss = nn.CrossEntropyLoss()
    device = torch.device(hy_params["device"])
    model = model.to(device)

    # --- EWC state (history across chunks) ---
    theta_stars = []
    fishers = []

    for epoch in range(hy_params['epochs']):
        for Nidx, N in enumerate(Niters):
            # ---- optimizer (your original structure) ----
            model.train()

            # ---- get current training window (your original logic) ----
            Xb, Yb_oh, y_target = X_train[N - dN:N], Y_train_oh[N - dN:N], y_train[N - dN:N]
            Xb = torch.as_tensor(X_train[N - dN:N], device=device, dtype=torch.float32)
            Yb_oh = torch.as_tensor(Yb_oh, device=hy_params['device'], dtype=Xb.dtype)
            y_target = torch.as_tensor(y_target, device=hy_params['device'], dtype=torch.long)

            # ---- SGD steps ----
            for start in range(0, Xb.shape[0], hy_params['sgd_batch_size']):
                end = min(start + hy_params['sgd_batch_size'], Xb.shape[0])
                x_mb = Xb[start:end]
                y_mb = Yb_oh[start:end]  # one-hot (float) for MSE case
                y_mb_target = y_target[start:end]  # LongTensor [B]

                # 1) classification forward (oracle gate during training if gating enabled)
                if getattr(model, "use_gating", False):
                    logits = model(x_mb, gate_ids=y_mb_target)
                else:
                    logits = model(x_mb)  # or model(x_mb) if baseline model

                # 1a) classification loss
                if hy_params['criterion'] == 'mse':
                    loss_cls = mse_loss(logits, y_mb)
                elif hy_params['criterion'] == 'crossentropy':
                    loss_cls = crossentropy_loss(logits, y_mb_target)
                else:
                    raise ValueError("incorrect criterion type")

                # --- add EWC penalty if available ---
                if len(fishers) > 0 and len(theta_stars) > 0:
                    loss = loss_cls + hy_params["ewc_lambda"] * ewc_penalty_multi_tasks(model, fishers, theta_stars)
                else:
                    loss = loss_cls
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"FATAL: NaN loss detected at step {N}. Terminating run.")
                    break
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()

            # ---- after training this chunk: update EWC state history ----
            if (N % hy_params.get("ewc_update_every", 1)) == 0 and hy_params['ewc_lambda'] > 0:
                theta_star_new = snapshot_params(model)  # store on same device as model
                fisher_new = estimate_fisher_diag(
                    model,
                    Xb, y_target,
                    device=device,
                    criterion=hy_params["criterion"],
                    fisher_n_samples=hy_params.get("ewc_fisher_n", None),
                )
                theta_stars.append(theta_star_new)
                fishers.append(fisher_new)

            # ---- evaluation ----
            model.eval()
            # Train metric: oracle gating makes most sense if gating is enabled
            tr_gate_mode = gate_mode if getattr(model, "use_gating", False) else "none"
            tr_acc, tr_mse = accuracy_cnn(
                model, X_train[:N], y_train[:N],
                device=hy_params['device'],
                gate_mode=tr_gate_mode)
            # Test metric: depends on gate_mode
            te_gate_mode = gate_mode if getattr(model, "use_gating", False) else "none"
            if hy_params['train_order_type'] == 'class_incremental':
                X_test_sub, y_test_sub = incre_seen_test_subset(y_train[:N], X_test, y_test)
                te_acc, te_mse = accuracy_cnn(
                    model, X_test_sub, y_test_sub,
                    device=hy_params['device'],
                    gate_mode=te_gate_mode)  # "infer" uses Option A; "oracle" uses true labels; "none" baseline
            elif hy_params['train_order_type'] == 'random':
                te_acc, te_mse = accuracy_cnn(
                    model, X_test, y_test,
                    device=hy_params['device'],
                    gate_mode=te_gate_mode
                )
            elif hy_params['train_order_type'] == 'task_incremental':
                total_tasks = 5
                te_samples_per_task = len(X_test) // total_tasks
                te_task_ids = torch.arange(len(X_test)) // te_samples_per_task
                te_acc, te_mse = accuracy_cnn(
                    model, X_test, y_test,
                    device=hy_params['device'],
                    gate_mode=te_gate_mode,
                    true_gate_ids=te_task_ids,
                )
            else:
                raise ValueError("Train order type error")

            train_accs.append(tr_acc); test_accs.append(te_acc)
            train_mses.append(tr_mse); test_mses.append(te_mse)
        train_accs_epochs.append(train_accs[-1]); test_accs_epochs.append(test_accs[-1])

    return train_accs_epochs, test_accs_epochs


#####################################################################
#-------------------------------CORe50------------------------------#
#####################################################################
def expand_core50_data_multi_epoch(data, task_classes, num_tr_per_class, num_epochs=3):
    # Detect if input is numpy, convert to tensor for efficient manipulation
    is_numpy = isinstance(data, np.ndarray)
    tensor = torch.as_tensor(data)

    repeated_blocks = []
    seen_samples = 0

    # We want to repeat the tensor along the 0th dimension (samples)
    # but keep all other dimensions (C, H, W) exactly the same.
    # If data is 4D (S, C, H, W), repeat_dims will be (num_epochs, 1, 1, 1)
    repeat_dims = [num_epochs] + [1] * (tensor.ndim - 1)

    for num_classes_in_task in task_classes:
        # 1. Calculate how many samples belong to the current task
        task_samples = num_classes_in_task * num_tr_per_class
        end_idx = seen_samples + task_samples

        # 2. Extract current task's block
        task_block = tensor[seen_samples:end_idx]

        # 3. Repeat the block for the number of epochs
        # This duplicates the block sequentially (Epoch 1 data, Epoch 2 data, etc.)
        expanded_task_block = task_block.repeat(*repeat_dims)

        repeated_blocks.append(expanded_task_block)

        # 4. Update the starting index for the next task
        seen_samples = end_idx

    # Concatenate all the expanded task blocks into one final sequential tensor
    final_tensor = torch.cat(repeated_blocks, dim=0).contiguous()

    # Convert back to numpy if the original input was numpy
    if is_numpy:
        return final_tensor.numpy()

    return final_tensor

def train_core50_van_sgd(
        model, X_train, y_train, X_test, y_test,
        hy_params, num_epochs, batch_size,
        num_tr_per_class, total_classes=50):
    # The NC scenario dictates 10 classes in the first batch, 5 in the rest
    device = hy_params['device']
    task_classes = [10, 5, 5, 5, 5, 5, 5, 5, 5]

    train_accs, test_accs = [], []

    seen_train_samples = 0
    opt = optimizer(hy_params, model)
    crossentropy_loss = nn.CrossEntropyLoss()

    for task_id, num_classes_in_task in enumerate(task_classes):
        print(f"--- Starting Task {task_id + 1}/{len(task_classes)} ({num_classes_in_task} classes) ---")

        # 1. Define the slice indices for the current training task
        task_train_samples = num_classes_in_task * num_tr_per_class
        start_tr = seen_train_samples
        end_tr = seen_train_samples + task_train_samples

        # 2. Extract current task's training data sequentially
        X_task = X_train[start_tr:end_tr]
        y_task = y_train[start_tr:end_tr]

        # Update seen training samples counter
        seen_train_samples += task_train_samples

        # 3. Multi-epoch training on the current batch (task)
        model.train()
        for epoch in range(num_epochs):
            # Shuffle indices to mix classes within the current task
            permutation = torch.randperm(X_task.shape[0])

            for i in range(0, X_task.shape[0], batch_size):
                indices = permutation[i:i + batch_size]

                # --- DEFENSIVE TENSOR CASTING ---
                # as_tensor handles both raw numpy arrays and existing tensors smoothly
                x_mb = torch.as_tensor(X_task[indices], dtype=torch.float32, device=device)
                y_mb_target = torch.as_tensor(y_task[indices], dtype=torch.int64, device=device)

                # Generate one-hot version for MSE, ensuring it is a float tensor
                y_mb = F.one_hot(y_mb_target, num_classes=total_classes).float()

                opt.zero_grad(set_to_none=True)

                # Baseline model forward pass
                logits = model(x_mb)

                # --- LOSS ROUTING ---
                if hy_params['criterion'] == 'mse':
                    loss_cls = mse_loss(logits, y_mb)
                elif hy_params['criterion'] == 'crossentropy':
                    loss_cls = crossentropy_loss(logits, y_mb_target)
                else:
                    raise ValueError("incorrect criterion type")

                # NaN/Inf Safety Check
                if torch.isnan(loss_cls) or torch.isinf(loss_cls):
                    print("FATAL: NaN loss detected")
                    break
                loss_cls.backward()
                opt.step()

        # 4. Evaluation Phase - USING FULL DATASETS
        model.eval()
        with torch.no_grad():

            # Use the FULL X_train and y_train
            tr_acc, tr_mse = accuracy_cnn(
                model, X_train, y_train,
                device=device, gate_mode="none", true_gate_ids=None
            )
            # Use the FULL X_test and y_test to match CORe50 Option 2
            te_acc, te_mse = accuracy_cnn(
                model, X_test, y_test,
                device=device, gate_mode="none", true_gate_ids=None
            )

            train_accs.append(tr_acc)
            test_accs.append(te_acc)

            print(f"Task {task_id + 1} | Full Train Acc: {tr_acc*100:.2f}% | Full Test Acc: {te_acc*100:.2f}%")

    return np.array(train_accs), np.array(test_accs)


def train_core50_van_sgd_ewc(
        model, X_train, y_train, X_test, y_test,
        hy_params, num_epochs, batch_size,
        num_tr_per_class, total_classes=50):
    device = hy_params['device']
    task_classes = [10, 5, 5, 5, 5, 5, 5, 5, 5]
    train_accs, test_accs = [], []
    seen_train_samples = 0
    opt = optimizer(hy_params, model)
    crossentropy_loss = nn.CrossEntropyLoss()

    theta_stars = []
    fishers = []
    ewc_lambda = float(hy_params.get("ewc_lambda", 0.0))

    for task_id, num_classes_in_task in enumerate(task_classes):
        print(f"--- Starting Task {task_id + 1}/{len(task_classes)} ({num_classes_in_task} classes, EWC) ---")
        task_train_samples = num_classes_in_task * num_tr_per_class
        start_tr = seen_train_samples
        end_tr = seen_train_samples + task_train_samples
        X_task = X_train[start_tr:end_tr]
        y_task = y_train[start_tr:end_tr]
        seen_train_samples += task_train_samples

        model.train()
        for epoch in range(num_epochs):
            permutation = torch.randperm(X_task.shape[0])
            for i in range(0, X_task.shape[0], batch_size):
                indices = permutation[i:i + batch_size]
                x_mb = torch.as_tensor(X_task[indices], dtype=torch.float32, device=device)
                y_mb_target = torch.as_tensor(y_task[indices], dtype=torch.int64, device=device)
                y_mb = F.one_hot(y_mb_target, num_classes=total_classes).float()

                opt.zero_grad(set_to_none=True)
                logits = model(x_mb)

                if hy_params['criterion'] == 'mse':
                    loss_cls = mse_loss(logits, y_mb)
                elif hy_params['criterion'] == 'crossentropy':
                    loss_cls = crossentropy_loss(logits, y_mb_target)
                else:
                    raise ValueError("incorrect criterion type")

                if ewc_lambda > 0 and len(fishers) > 0 and len(theta_stars) > 0:
                    loss = loss_cls + ewc_lambda * ewc_penalty_multi_tasks(model, fishers, theta_stars)
                else:
                    loss = loss_cls

                if torch.isnan(loss) or torch.isinf(loss):
                    print("FATAL: NaN loss detected")
                    break
                loss.backward()
                opt.step()

        model.eval()
        with torch.no_grad():
            tr_acc, tr_mse = accuracy_cnn(
                model, X_train, y_train,
                device=device, gate_mode="none", true_gate_ids=None
            )
            te_acc, te_mse = accuracy_cnn(
                model, X_test, y_test,
                device=device, gate_mode="none", true_gate_ids=None
            )
            train_accs.append(tr_acc)
            test_accs.append(te_acc)
            print(f"Task {task_id + 1} | Full Train Acc: {tr_acc*100:.2f}% | Full Test Acc: {te_acc*100:.2f}%")

        if ewc_lambda > 0:
            theta_stars.append(snapshot_params(model))
            fishers.append(
                estimate_fisher_diag(
                    model,
                    X_task,
                    y_task,
                    device=device,
                    criterion=hy_params["criterion"],
                    fisher_n_samples=hy_params.get("ewc_fisher_n", None),
                )
            )

    return np.array(train_accs), np.array(test_accs)


def train_core50_van_sgd_cumulative(
        model, X_train, y_train, X_test, y_test,
        hy_params, num_epochs, batch_size,
        num_tr_per_class, total_classes=50):
    device = hy_params['device']
    task_classes = [10, 5, 5, 5, 5, 5, 5, 5, 5]

    train_accs, test_accs = [], []
    seen_train_samples = 0
    opt = optimizer(hy_params, model)
    crossentropy_loss = nn.CrossEntropyLoss()

    for task_id, num_classes_in_task in enumerate(task_classes):
        print(f"--- Starting Task {task_id + 1}/{len(task_classes)} ({num_classes_in_task} classes, cumulative) ---")
        task_train_samples = num_classes_in_task * num_tr_per_class
        seen_train_samples += task_train_samples

        # cumulative window over all seen tasks
        X_task = X_train[:seen_train_samples]
        y_task = y_train[:seen_train_samples]

        model.train()
        for epoch in range(num_epochs):
            permutation = torch.randperm(X_task.shape[0])

            for i in range(0, X_task.shape[0], batch_size):
                indices = permutation[i:i + batch_size]
                x_mb = torch.as_tensor(X_task[indices], dtype=torch.float32, device=device)
                y_mb_target = torch.as_tensor(y_task[indices], dtype=torch.int64, device=device)
                y_mb = F.one_hot(y_mb_target, num_classes=total_classes).float()

                opt.zero_grad(set_to_none=True)
                logits = model(x_mb)

                if hy_params['criterion'] == 'mse':
                    loss_cls = mse_loss(logits, y_mb)
                elif hy_params['criterion'] == 'crossentropy':
                    loss_cls = crossentropy_loss(logits, y_mb_target)
                else:
                    raise ValueError("incorrect criterion type")

                if torch.isnan(loss_cls) or torch.isinf(loss_cls):
                    print("FATAL: NaN loss detected")
                    break
                loss_cls.backward()
                opt.step()

        model.eval()
        with torch.no_grad():
            tr_acc, tr_mse = accuracy_cnn(
                model, X_train, y_train,
                device=device, gate_mode="none", true_gate_ids=None
            )
            te_acc, te_mse = accuracy_cnn(
                model, X_test, y_test,
                device=device, gate_mode="none", true_gate_ids=None
            )
            train_accs.append(tr_acc)
            test_accs.append(te_acc)
            print(f"Task {task_id + 1} | Full Train Acc: {tr_acc*100:.2f}% | Full Test Acc: {te_acc*100:.2f}%")

    return np.array(train_accs), np.array(test_accs)

def dumb_train_core50_label_correction_adap_kernel(model, X_train, y_train, X_test, y_test, hy_params, sgd_on_eta=0.01, sgd_on_gm=100, online_cor_type='batch_gpu', num_classes=50):
    adap_kernel_times = hy_params['adap_kernel_times']
    device, kernel_tp, z_type, frames_per_session = hy_params['device'], hy_params['kernel_tp'], hy_params['z_type'], hy_params['frames_per_session']
    num_epochs, z_mini_bz = hy_params['epochs'], hy_params['z_mini_bz']
    task_classes, num_tr_per_class = [10, 5, 5, 5, 5, 5, 5, 5, 5], 8*frames_per_session

    start_task_boundaries, end_task_boundaries = [], []
    end_index = 0
    for num_classes in task_classes:
        # Samples for this task across ALL its epochs combined
        start_task_boundaries.append(end_index)
        multi_epoch_task_samples = num_classes * num_tr_per_class * num_epochs
        end_index += multi_epoch_task_samples
        end_task_boundaries.append(end_index)

    model = model.to(device)
    opt = torch.optim.SGD(
        model.parameters(),
        lr=sgd_on_eta,
        momentum=hy_params['sgd_momentum'],
        weight_decay=hy_params['sgd_weight_decay']
    )
    # Expand the datasets to multi-epoch (Task1-Task1-Task1...)
    X_train_multi = expand_core50_data_multi_epoch(X_train, task_classes, num_tr_per_class, num_epochs=num_epochs)
    y_train_multi = expand_core50_data_multi_epoch(y_train, task_classes, num_tr_per_class, num_epochs=num_epochs)
    Y_train_oh_multi = one_hot(y_train_multi, hy_params['num_classes'])

    # initialization of kernel and Z_iter_oh
    if adap_kernel_times > 1:
        # Use only Task 1's multi-epoch data
        init_end_idx = end_task_boundaries[0]
    else:
        # Use ALL samples across all tasks and epochs
        init_end_idx = end_task_boundaries[-1]
        # (Note: using `init_end_idx = None` also works perfectly here for slicing)
    K_train_multi, K_tr_te_multi = kernel_type(model, X_train_multi[:init_end_idx], X_test, device, kernel_tp, hy_params)
    Z_iter_oh_multi = compute_Z_iter(K_train_multi, Y_train_oh_multi[:init_end_idx], sgd_on_gm, sgd_on_eta, hy_params['on_cor_bs'], online_cor_type, z_type, mini_bz=z_mini_bz)

    steps, train_accs, test_accs, train_mses, test_mses = [], [], [], [], []

    for task_id, task_end_idx in enumerate(end_task_boundaries):
        task_start_idx = start_task_boundaries[task_id]
        steps.append(task_id+1)
        print("current task:", task_id+1)
        # ---- optimizer (your original structure) ----
        model.train()
        # ---- get current training window (your original logic) ----
        Xb, Zb_oh, y_target = X_train_multi[task_start_idx:task_end_idx], Z_iter_oh_multi[task_start_idx:task_end_idx], y_train_multi[task_start_idx:task_end_idx]
        Xb = torch.as_tensor(Xb, device=device, dtype=torch.float32)
        Zb_oh = torch.as_tensor(Zb_oh, device=hy_params['device'], dtype=Xb.dtype)

        # ---- SGD steps ----
        for start in range(0, Xb.shape[0], hy_params['sgd_batch_size']):
            end = min(start + hy_params['sgd_batch_size'], Xb.shape[0])
            x_mb = Xb[start:end]
            z_mb = Zb_oh[start:end]  # one-hot (float) for MSE case

            # Baseline model (no gating)
            logits = model(x_mb)
            # 1a) classification loss
            loss_cls = mse_loss(logits, z_mb)

            # --- NEW: NaN/Inf Safety Check ---
            if torch.isnan(loss_cls) or torch.isinf(loss_cls):
                print(f"FATAL: NaN loss detected . Terminating run.")
                # You can choose to 'break' the loop, or return early to move to the next seed
                break
            opt.zero_grad(set_to_none=True)
            loss_cls.backward()
            opt.step()

        # ---- evaluation ----
        model.eval()
        with torch.no_grad():

            # Use the FULL X_train and y_train
            tr_acc, tr_mse = accuracy_cnn(
                model, X_train, y_train,
                device=device, gate_mode="none", true_gate_ids=None
            )
            # Use the FULL X_test and y_test to match CORe50 Option 2
            te_acc, te_mse = accuracy_cnn(
                model, X_test, y_test,
                device=device, gate_mode="none", true_gate_ids=None
            )

            train_accs.append(tr_acc)
            test_accs.append(te_acc)

            print(f"Task {task_id + 1} | Full Train Acc: {tr_acc*100:.2f}% | Full Test Acc: {te_acc*100:.2f}%")

        train_accs.append(tr_acc); test_accs.append(te_acc)

        # update kernel
        model.eval()
        if adap_kernel_times > 1 and (task_id + 1) < len(end_task_boundaries):
            print(f"Updating the kernel")
            next_update_end = end_task_boundaries[task_id+1]
            # update of the Z_iter_oh and Kernel
            K_train_multi, K_tr_te_multi = kernel_type(model, X_train_multi[:next_update_end], X_test, device,
                                                       kernel_tp, hy_params)
            Z_iter_oh_multi = compute_Z_iter(K_train_multi, Y_train_oh_multi[:next_update_end], sgd_on_gm, sgd_on_eta,
                                             hy_params['on_cor_bs'], online_cor_type, z_type, mini_bz=z_mini_bz)

    return np.array(train_accs), np.array(test_accs)


def efficient_train_core50_label_correction_adap_kernel(model, X_train, y_train, X_test, y_test, hy_params, sgd_on_eta=0.01, sgd_on_gm=100, online_cor_type='batch_gpu'):
    adap_kernel_times = hy_params['adap_kernel_times']
    device, kernel_tp, z_type, frames_per_session = hy_params['device'], hy_params['kernel_tp'], hy_params['z_type'], hy_params['frames_per_session']
    num_epochs, z_mini_bz = hy_params['epochs'], hy_params['z_mini_bz']
    task_classes, num_tr_per_class = [10, 5, 5, 5, 5, 5, 5, 5, 5], 8*frames_per_session

    start_task_boundaries, end_task_boundaries = [], []
    end_index = 0
    for num_classes in task_classes:
        # Samples for this task across ALL its epochs combined
        start_task_boundaries.append(end_index)
        multi_epoch_task_samples = num_classes * num_tr_per_class * num_epochs
        end_index += multi_epoch_task_samples
        end_task_boundaries.append(end_index)

    model = model.to(device)
    opt = torch.optim.SGD(
        model.parameters(),
        lr=sgd_on_eta,
        momentum=hy_params['sgd_momentum'],
        weight_decay=hy_params['sgd_weight_decay']
    )
    # Expand the datasets to multi-epoch (Task1-Task1-Task1...)
    X_train_multi = expand_core50_data_multi_epoch(X_train, task_classes, num_tr_per_class, num_epochs=num_epochs)
    y_train_multi = expand_core50_data_multi_epoch(y_train, task_classes, num_tr_per_class, num_epochs=num_epochs)
    Y_train_oh_multi = one_hot(y_train_multi, hy_params['num_classes'])

    # initialization of kernel and Z_iter_oh
    if adap_kernel_times > 1:
        # Use all epochs of Task 1's data
        init_end_idx = end_task_boundaries[0]
        K_train_multi, K_tr_te_multi = kernel_type(model, X_train_multi[:init_end_idx], X_test, device, kernel_tp,
                                                   hy_params)
        Z_iter_oh_multi = compute_Z_iter(K_train_multi, Y_train_oh_multi[:init_end_idx], sgd_on_gm, sgd_on_eta,
                                         hy_params['on_cor_bs'], online_cor_type, z_type, mini_bz=z_mini_bz)

        # Store the corrected labels ready for Task 1
        current_Zb_oh = Z_iter_oh_multi[:init_end_idx]
    else:
        init_end_idx = end_task_boundaries[-1]
        K_train_multi, K_tr_te_multi = kernel_type(model, X_train_multi[:init_end_idx], X_test, device, kernel_tp, hy_params)
        Z_iter_oh_multi = compute_Z_iter(K_train_multi, Y_train_oh_multi[:init_end_idx], sgd_on_gm, sgd_on_eta, hy_params['on_cor_bs'], online_cor_type, z_type, mini_bz=z_mini_bz)

    steps, train_accs, test_accs, train_mses, test_mses = [], [], [], [], []

    for task_id, task_end_idx in enumerate(end_task_boundaries):
        task_start_idx = start_task_boundaries[task_id]
        steps.append(task_id+1)
        print("current task:", task_id+1)
        # ---- optimizer (your original structure) ----
        model.train()
        # ---- get current training window (your original logic) ----
        Xb, y_target = X_train_multi[task_start_idx:task_end_idx], y_train_multi[task_start_idx:task_end_idx]
        if adap_kernel_times > 1:
            # Use the labels we prepared in the previous step!
            Zb_oh = current_Zb_oh
        else:
            Zb_oh = Z_iter_oh_multi[task_start_idx:task_end_idx]
        Xb = torch.as_tensor(Xb, device=device, dtype=torch.float32)
        Zb_oh = torch.as_tensor(Zb_oh, device=hy_params['device'], dtype=Xb.dtype)

        # ---- SGD steps ----
        for start in range(0, Xb.shape[0], hy_params['sgd_batch_size']):
            end = min(start + hy_params['sgd_batch_size'], Xb.shape[0])
            x_mb = Xb[start:end]
            z_mb = Zb_oh[start:end]  # one-hot (float) for MSE case

            # Baseline model (no gating)
            logits = model(x_mb)
            # 1a) classification loss
            loss_cls = mse_loss(logits, z_mb)

            # --- NEW: NaN/Inf Safety Check ---
            if torch.isnan(loss_cls) or torch.isinf(loss_cls):
                print(f"FATAL: NaN loss detected . Terminating run.")
                # You can choose to 'break' the loop, or return early to move to the next seed
                break
            opt.zero_grad(set_to_none=True)
            loss_cls.backward()
            opt.step()

        # ---- evaluation ----
        model.eval()
        with torch.no_grad():

            # Use the FULL X_train and y_train
            tr_acc, tr_mse = accuracy_cnn(
                model, X_train, y_train,
                device=device, gate_mode="none", true_gate_ids=None
            )
            # Use the FULL X_test and y_test to match CORe50 Option 2
            te_acc, te_mse = accuracy_cnn(
                model, X_test, y_test,
                device=device, gate_mode="none", true_gate_ids=None
            )

            train_accs.append(tr_acc)
            test_accs.append(te_acc)

            print(f"Task {task_id + 1} | Full Train Acc: {tr_acc*100:.2f}% | Full Test Acc: {te_acc*100:.2f}%")

        # update kernel
        model.eval()
        if adap_kernel_times > 1 and (task_id + 1) < len(end_task_boundaries):
            print(f"Updating the kernel (History: 1 Epoch / Current: All Epochs)")
            next_task = task_id + 1

            subset_indices = []

            # A. Get exactly 1 epoch for all PREVIOUS tasks (0 to task_id)
            for past_t in range(next_task):
                samples_in_one_epoch = task_classes[past_t] * num_tr_per_class
                epoch1_start = start_task_boundaries[past_t]
                epoch1_end = epoch1_start + samples_in_one_epoch

                # Append this specific chunk of indices
                subset_indices.extend(range(epoch1_start, epoch1_end))

            # B. Get ALL epochs for the NEXT task (the labels we are trying to correct)
            next_task_start = start_task_boundaries[next_task]
            next_task_end = end_task_boundaries[next_task]
            subset_indices.extend(range(next_task_start, next_task_end))

            # Extract the subset data (works for both numpy arrays and PyTorch tensors)
            X_subset = X_train_multi[subset_indices]
            Y_subset = Y_train_oh_multi[subset_indices]

            # update of the Z_iter_oh and Kernel using the reduced subset
            K_train_multi, K_tr_te_multi = kernel_type(model, X_subset, X_test, device, kernel_tp, hy_params)

            Z_iter_subset = compute_Z_iter(K_train_multi, Y_subset, sgd_on_gm, sgd_on_eta,
                                           hy_params['on_cor_bs'], online_cor_type, z_type,
                                           mini_bz=z_mini_bz)

            # --- THE CRITICAL STEP ---
            # The corrected labels for the NEXT task are located at the very end
            # of our Z_iter_subset. We extract exactly that many trailing elements.
            num_next_task_samples = next_task_end - next_task_start
            current_Zb_oh = Z_iter_subset[-num_next_task_samples:]

    return np.array(train_accs), np.array(test_accs)


def full_efficient_train_core50_label_correction_adap_kernel(
        model, X_train, y_train, X_test, y_test, hy_params, sgd_on_eta=0.01, sgd_on_gm=100, online_cor_type='batch_gpu'):
    device = hy_params['device']
    kernel_tp = hy_params['kernel_tp']
    z_type = hy_params['z_type']
    frames_per_session = hy_params['frames_per_session']
    num_epochs, z_mini_bz = hy_params['epochs'], hy_params['z_mini_bz']
    task_classes, num_tr_per_class = [10, 5, 5, 5, 5, 5, 5, 5, 5], 8 * frames_per_session

    start_task_boundaries, end_task_boundaries = [], []
    end_index = 0
    for num_classes in task_classes:
        start_task_boundaries.append(end_index)
        multi_epoch_task_samples = num_classes * num_tr_per_class * num_epochs
        end_index += multi_epoch_task_samples
        end_task_boundaries.append(end_index)

    model = model.to(device)
    opt = torch.optim.SGD(
        model.parameters(),
        lr=sgd_on_eta,
        momentum=hy_params['sgd_momentum'],
        weight_decay=hy_params['sgd_weight_decay']
    )
    crossentropy_loss = nn.CrossEntropyLoss()

    X_train_multi = expand_core50_data_multi_epoch(X_train, task_classes, num_tr_per_class, num_epochs=num_epochs)
    y_train_multi = expand_core50_data_multi_epoch(y_train, task_classes, num_tr_per_class, num_epochs=num_epochs)
    Y_train_oh_multi = one_hot(y_train_multi, hy_params['num_classes'])

    train_accs, test_accs = [], []

    for task_id in range(len(task_classes)):
        samples_per_epoch = task_classes[task_id] * num_tr_per_class
        task_start = start_task_boundaries[task_id]
        print(f"--- Task {task_id + 1}/{len(task_classes)} (epoch-wise label correction) ---")

        for epoch_idx in range(num_epochs):
            epoch_abs_start = task_start + epoch_idx * samples_per_epoch
            epoch_abs_end = epoch_abs_start + samples_per_epoch

            subset_indices = []
            if task_id > 0:
                for past_t in range(task_id):
                    sp = task_classes[past_t] * num_tr_per_class
                    pst = start_task_boundaries[past_t]
                    last_ep_start = pst + (num_epochs - 1) * sp
                    subset_indices.extend(range(last_ep_start, last_ep_start + sp))
            subset_indices.extend(range(epoch_abs_start, epoch_abs_end))

            X_subset = X_train_multi[subset_indices]
            Y_subset = Y_train_oh_multi[subset_indices]

            model.eval()
            K_train_multi, _K_tr_te = kernel_type(model, X_subset, X_test, device, kernel_tp, hy_params)
            Z_iter_subset = compute_Z_iter(
                K_train_multi,
                Y_subset,
                sgd_on_gm,
                sgd_on_eta,
                hy_params['on_cor_bs'],
                online_cor_type,
                z_type,
                mini_bz=z_mini_bz,
            )

            num_curr = epoch_abs_end - epoch_abs_start
            Z_curr = Z_iter_subset[-num_curr:]
            Z_curr_t = torch.as_tensor(Z_curr, device=device, dtype=torch.float32)
            X_epoch_t = torch.as_tensor(
                X_train_multi[epoch_abs_start:epoch_abs_end],
                device=device,
                dtype=torch.float32,
            )

            model.train()
            for start in range(0, X_epoch_t.shape[0], hy_params['sgd_batch_size']):
                end = min(start + hy_params['sgd_batch_size'], X_epoch_t.shape[0])
                x_mb = X_epoch_t[start:end]
                z_mb = Z_curr_t[start:end]
                y_slice = y_train_multi[epoch_abs_start + start: epoch_abs_start + end]
                y_mb_target = torch.as_tensor(y_slice, device=device, dtype=torch.long)

                logits = model(x_mb)
                if hy_params['criterion'] == 'mse':
                    loss_cls = mse_loss(logits, z_mb)
                elif hy_params['criterion'] == 'crossentropy':
                    loss_cls = crossentropy_loss(logits, y_mb_target)
                else:
                    raise ValueError("incorrect criterion type")

                if torch.isnan(loss_cls) or torch.isinf(loss_cls):
                    print(f"FATAL: NaN loss at task {task_id + 1} epoch {epoch_idx + 1}. Terminating run.")
                    break
                opt.zero_grad(set_to_none=True)
                loss_cls.backward()
                opt.step()

        model.eval()
        with torch.no_grad():
            tr_acc, tr_mse = accuracy_cnn(
                model, X_train, y_train,
                device=device, gate_mode="none", true_gate_ids=None
            )
            te_acc, te_mse = accuracy_cnn(
                model, X_test, y_test,
                device=device, gate_mode="none", true_gate_ids=None
            )
            train_accs.append(tr_acc)
            test_accs.append(te_acc)
            print(f"Task {task_id + 1} | Full Train Acc: {tr_acc*100:.2f}% | Full Test Acc: {te_acc*100:.2f}%")

    return np.array(train_accs), np.array(test_accs)