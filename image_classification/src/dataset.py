import torch
from torchvision import datasets, transforms
import numpy as np
import jax.numpy as jnp
import torchvision.transforms as T
import pickle as pkl
import re
from collections import defaultdict

def torch_to_jax_nhwc(X_torch):
    # X_torch: torch.Tensor (N,1,28,28)
    X = X_torch.detach().cpu().numpy().astype(np.float32)
    return jnp.array(np.transpose(X, (0, 2, 3, 1)))  # (N,28,28,1)

def make_order_multiclass(y: np.ndarray, seed: int, order_type="random", n_classes=10) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = len(y)

    if order_type == "random":
        return rng.permutation(n)

    if order_type == "class_incremental":
        chunks = []
        for c in range(n_classes):
            idx = np.where(y == c)[0]
            idx = rng.permutation(idx)  # shuffle within class
            chunks.append(idx)
        return np.concatenate(chunks)

    raise ValueError(f"Unknown order_type={order_type}. Use 'random' or 'class_incremental'.")

def whiten_like_jax(X: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    mean = X.mean(dim=0, keepdim=True)                    # (1,1,28,28)
    std  = X.std(dim=0, keepdim=True, unbiased=False)     # ddof=0 like jnp.std default
    return (X - mean) / (std + eps)


def load_mnist_subset_whitened_old(
    n_train: int,
    n_test: int,
    root: str = "./data",
    seed_select: int = 0,
    eps: float = 1e-6,
    whiten_ds: bool = True,
    device: str = "cpu",
):
    tfm = transforms.ToTensor()
    train_ds = datasets.MNIST(root=root, train=True,  download=True, transform=tfm)
    test_ds  = datasets.MNIST(root=root, train=False, download=True, transform=tfm)

    # fixed subset selection
    base_rng = np.random.default_rng(seed_select)
    tr_idx = base_rng.choice(len(train_ds), size=n_train, replace=False)
    te_idx = base_rng.choice(len(test_ds),  size=n_test,  replace=False)

    # stack tensors (N,1,28,28)
    X_train_all = torch.stack([train_ds[i][0] for i in tr_idx], dim=0).to(device)
    y_train_all = np.array([train_ds[i][1] for i in tr_idx], dtype=np.int64)

    X_test = torch.stack([test_ds[i][0] for i in te_idx], dim=0).to(device)
    y_test = np.array([test_ds[i][1] for i in te_idx], dtype=np.int64)

    # whitening
    if whiten_ds:
        X_train_all = whiten_like_jax(X_train_all, eps=eps)
        X_test      = whiten_like_jax(X_test, eps=eps)

    return X_train_all, y_train_all, X_test, y_test, tr_idx, te_idx

def load_mnist_subset_whitened(n_train: int, n_test: int, root: str = "./data", seed_select: int = 0, eps: float = 1e-6, whiten_ds: bool = True, device: str = "cpu"):
    n_classes = 10
    assert n_train >= n_classes and n_test >= n_classes
    rng = np.random.default_rng(seed_select)

    tfm = transforms.ToTensor()
    train_ds = datasets.MNIST(root=root, train=True, download=True, transform=tfm)
    test_ds  = datasets.MNIST(root=root, train=False, download=True, transform=tfm)

    y_train_full = train_ds.targets.cpu().numpy()
    y_test_full  = test_ds.targets.cpu().numpy()

    ntr = n_train // n_classes
    nte = n_test  // n_classes
    if ntr * n_classes != n_train or nte * n_classes != n_test:
        print(f"[warn] n_train or n_test not divisible by 10. "
              f"Using {ntr} per class for train and {nte} per class for test "
              f"(total {ntr*n_classes}, {nte*n_classes}).")
    tr_chunks, te_chunks = [], []
    for c in range(n_classes):
        tr_pool = np.where(y_train_full == c)[0]
        te_pool = np.where(y_test_full == c)[0]
        tr_sel = rng.choice(tr_pool, size=ntr, replace=False)
        te_sel = rng.choice(te_pool, size=nte, replace=False)
        tr_chunks.append(tr_sel)
        te_chunks.append(te_sel)
    tr_idx = np.concatenate(tr_chunks)
    te_idx = np.concatenate(te_chunks)

    rng.shuffle(tr_idx)
    rng.shuffle(te_idx)

    X_train_all = torch.stack([train_ds[i][0] for i in tr_idx], dim=0).to(device)
    y_train_all = np.array([train_ds[i][1] for i in tr_idx], dtype=np.int64)

    X_test = torch.stack([test_ds[i][0] for i in te_idx], dim=0).to(device)
    y_test = np.array([test_ds[i][1] for i in te_idx], dtype=np.int64)
    # whitening
    if whiten_ds:
        X_train_all = whiten_like_jax(X_train_all, eps=eps)
        X_test      = whiten_like_jax(X_test, eps=eps)
    return X_train_all, y_train_all, X_test, y_test, tr_idx, te_idx


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2470, 0.2435, 0.2616)

def load_cifar10_subset(
        n_train: int,
        n_test: int,
        root: str = "./data",
        seed_select: int = 0,
        normalize_cifar10: bool = True,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
):
    """
    Returns perfectly class-balanced subsets of CIFAR-10.
    """
    tfm_list = [transforms.ToTensor()]

    # Standard CIFAR-10 normalization
    if normalize_cifar10:
        tfm_list.append(transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD))

    tfm = transforms.Compose(tfm_list)

    train_ds = datasets.CIFAR10(root=root, train=True, download=True, transform=tfm)
    test_ds = datasets.CIFAR10(root=root, train=False, download=True, transform=tfm)

    rng = np.random.default_rng(seed_select)

    n_tr_per_class = n_train // 10
    n_te_per_class = n_test // 10

    y_tr_full = np.array(train_ds.targets)
    y_te_full = np.array(test_ds.targets)

    tr_idx_list = []
    te_idx_list = []

    # Loop through all 10 classes and sample equally
    for c in range(10):
        c_tr_idx = np.where(y_tr_full == c)[0]
        c_te_idx = np.where(y_te_full == c)[0]

        sampled_c_tr = rng.choice(c_tr_idx, size=n_tr_per_class, replace=False)
        sampled_c_te = rng.choice(c_te_idx, size=n_te_per_class, replace=False)

        tr_idx_list.extend(sampled_c_tr)
        te_idx_list.extend(sampled_c_te)

    tr_idx = np.array(tr_idx_list)
    te_idx = np.array(te_idx_list)
    rng.shuffle(tr_idx)
    rng.shuffle(te_idx)

    X_train_all = torch.stack([train_ds[i][0] for i in tr_idx], dim=0).to(device=device, dtype=dtype)
    y_train_all = np.array([train_ds[i][1] for i in tr_idx], dtype=np.int64)

    X_test = torch.stack([test_ds[i][0] for i in te_idx], dim=0).to(device=device, dtype=dtype)
    y_test = np.array([test_ds[i][1] for i in te_idx], dtype=np.int64)

    return X_train_all, y_train_all, X_test, y_test, tr_idx, te_idx


def load_ds(n_train, n_test, root, seed_select, device, hy_params):
    if hy_params['ds_type']=='mnist':
        X_train_all, y_train_all, X_test, y_test, tr_idx, te_idx = load_mnist_subset_whitened(n_train, n_test, root, seed_select, device=device)
    else:
        X_train_all, y_train_all, X_test, y_test, tr_idx, te_idx = load_cifar10_subset(n_train, n_test, root, seed_select, normalize_cifar10=True, device=device)

    return X_train_all, y_train_all, X_test, y_test, tr_idx, te_idx


def format_split_binary_tasks(X_train_all, y_train_all, X_test, y_test, seed=42):
    # 1. Randomly group the 10 classes into 5 pairs based on the seed
    rng = np.random.default_rng(seed)
    all_classes = np.arange(10)
    rng.shuffle(all_classes)
    task_pairs = all_classes.reshape(5, 2)

    print("--- Split Task Sequence ---")
    for i, pair in enumerate(task_pairs):
        print(f"Task {i + 1}: Original Classes {pair[0]} vs {pair[1]} -> Mapped to 0 vs 1")
    print("------------------------------------")

    # 2. Prepare lists to hold the sorted, task-separated data
    new_X_train, new_y_train = [], []
    new_X_test, new_y_test = [], []

    # 3. Iterate through each task pair, extract the data, and relabel
    for class_0, class_1 in task_pairs:
        train_mask_0 = (y_train_all == class_0)
        train_mask_1 = (y_train_all == class_1)
        train_mask_task = train_mask_0 | train_mask_1

        X_tr_task = X_train_all[train_mask_task]
        y_tr_raw = y_train_all[train_mask_task]

        y_tr_binary = np.zeros_like(y_tr_raw)
        y_tr_binary[y_tr_raw == class_1] = 1

        n_train_samples = len(y_tr_binary)
        shuf_idx_tr = rng.permutation(n_train_samples) 
        X_tr_shuffled = X_tr_task[shuf_idx_tr]
        y_tr_shuffled = y_tr_binary[shuf_idx_tr]

        new_X_train.append(X_tr_shuffled)
        new_y_train.append(y_tr_shuffled)

        test_mask_0 = (y_test == class_0)
        test_mask_1 = (y_test == class_1)
        test_mask_task = test_mask_0 | test_mask_1

        X_te_task = X_test[test_mask_task]
        y_te_raw = y_test[test_mask_task]

        y_te_binary = np.zeros_like(y_te_raw)
        y_te_binary[y_te_raw == class_1] = 1

        n_test_samples = len(y_te_binary)
        shuf_idx_te = rng.permutation(n_test_samples)

        X_te_shuffled = X_te_task[shuf_idx_te]
        y_te_shuffled = y_te_binary[shuf_idx_te]

        new_X_test.append(X_te_shuffled)
        new_y_test.append(y_te_shuffled)

    X_train_seq = torch.cat(new_X_train, dim=0)
    y_train_seq = np.concatenate(new_y_train, axis=0)
    X_test_seq = torch.cat(new_X_test, dim=0)
    y_test_seq = np.concatenate(new_y_test, axis=0)

    return X_train_seq, y_train_seq, X_test_seq, y_test_seq, task_pairs


def format_split_10class_default_tasks(X_train_all, y_train_all, X_test, y_test, seed=42):
    task_pairs = np.array([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]])

    print("--- Standard Task-Incremental Sequence ---")
    for i, pair in enumerate(task_pairs):
        print(f"Task {i + 1}: Original Classes {pair[0]} vs {pair[1]} -> Mapped to 0 vs 1")
    print("------------------------------------------")

    new_X_train, new_y_train = [], []
    new_X_test, new_y_test = [], []

    rng = np.random.default_rng(seed)

    for class_0, class_1 in task_pairs:
        # --- Training Data Extraction & Relabeling ---
        train_mask = (y_train_all == class_0) | (y_train_all == class_1)

        X_tr_task = X_train_all[train_mask]
        y_tr_raw = y_train_all[train_mask]

        # Map to binary
        y_tr_binary = np.zeros_like(y_tr_raw)
        y_tr_binary[y_tr_raw == class_1] = 1

        # Safely Shuffle Training Task Block
        n_tr_samples = len(y_tr_binary)
        shuf_idx_tr = rng.permutation(n_tr_samples)

        new_X_train.append(X_tr_task[shuf_idx_tr])
        new_y_train.append(y_tr_binary[shuf_idx_tr])

        # --- Testing Data Extraction & Relabeling ---
        test_mask = (y_test == class_0) | (y_test == class_1)

        X_te_task = X_test[test_mask]
        y_te_raw = y_test[test_mask]

        # Map to binary
        y_te_binary = np.zeros_like(y_te_raw)
        y_te_binary[y_te_raw == class_1] = 1

        # Safely Shuffle Testing Task Block
        n_te_samples = len(y_te_binary)
        shuf_idx_te = rng.permutation(n_te_samples)

        new_X_test.append(X_te_task[shuf_idx_te])
        new_y_test.append(y_te_binary[shuf_idx_te])

    X_train_seq = torch.cat(new_X_train, dim=0)
    y_train_seq = np.concatenate(new_y_train, axis=0)
    X_test_seq = torch.cat(new_X_test, dim=0)
    y_test_seq = np.concatenate(new_y_test, axis=0)

    return X_train_seq, y_train_seq, X_test_seq, y_test_seq, task_pairs


def load_core50_subset_whitened(
        n_train=None,
        n_test=None,
        npz_path: str = "core50_imgs.npz",
        pkl_path: str = "paths.pkl",
        frames_per_session: int = 30,
        img_size: int = 64,
        seed_select: int = 0,
        device: str = "cpu"
):
    if n_train == None:
        n_train, n_test = 50*8*frames_per_session, 50*3*frames_per_session
    n_classes = 50
    assert n_train >= n_classes and n_test >= n_classes
    rng = np.random.default_rng(seed_select)

    # 1. Load Data
    print("Loading CORe50 Data...")
    imgs = np.load(npz_path)['x']
    with open(pkl_path, 'rb') as f:
        paths = pkl.load(f)

    y_full = np.zeros(len(paths), dtype=np.int64)
    sessions = np.zeros(len(paths), dtype=np.int64)
    sess_class_groups = defaultdict(list)

    for i, p in enumerate(paths):
        match = re.search(r's(\d+)/o(\d+)', p)
        if match:
            s = int(match.group(1))
            c = int(match.group(2)) - 1  # 0 to 49
            sessions[i] = s
            y_full[i] = c
            sess_class_groups[(s, c)].append(i)

    valid_indices = []
    for (s, c), idxs in sess_class_groups.items():
        if frames_per_session is not None and len(idxs) > frames_per_session:
            spaced_idxs = np.linspace(0, len(idxs) - 1, frames_per_session, dtype=int)
            sel_idxs = [idxs[k] for k in spaced_idxs]
        else:
            sel_idxs = idxs
        valid_indices.extend(sel_idxs)

    valid_indices = np.array(valid_indices)

    train_sessions = [1, 2, 4, 5, 6, 8, 9, 11]  # Added session 6
    test_sessions = [3, 7, 10]

    train_mask = np.isin(sessions, train_sessions)
    test_mask = np.isin(sessions, test_sessions)

    tr_global_idx = np.intersect1d(np.where(train_mask)[0], valid_indices)
    te_global_idx = np.intersect1d(np.where(test_mask)[0], valid_indices)

    categories = [list(range(cat * 5, (cat + 1) * 5)) for cat in range(10)]
    for cat in categories:
        rng.shuffle(cat)  # Shuffle objects within each category for randomness per seed

    batches = [[] for _ in range(9)]

    for cat_idx in range(10):
        batches[0].append(categories[cat_idx].pop())

    flat_remaining = [obj for cat in categories for obj in cat]
    rng.shuffle(flat_remaining)

    idx = 0
    for b in range(1, 9):
        batches[b] = flat_remaining[idx:idx + 5]
        idx += 5

    ntr = n_train // n_classes
    nte = n_test // n_classes

    tr_ordered_chunks = []
    te_ordered_chunks = []

    for batch_classes in batches:
        batch_tr_idx = []
        batch_te_idx = []

        for c in batch_classes:
            c_idx = np.where(y_full == c)[0]
            c_tr_pool = np.intersect1d(tr_global_idx, c_idx)
            c_te_pool = np.intersect1d(te_global_idx, c_idx)
            safe_ntr = min(ntr, len(c_tr_pool))
            batch_tr_idx.append(rng.choice(c_tr_pool, size=safe_ntr, replace=False))
            safe_nte = min(nte, len(c_te_pool))
            batch_te_idx.append(rng.choice(c_te_pool, size=safe_nte, replace=False))

        b_tr_flat = np.concatenate(batch_tr_idx)
        b_te_flat = np.concatenate(batch_te_idx)
        rng.shuffle(b_tr_flat)
        rng.shuffle(b_te_flat)

        tr_ordered_chunks.append(b_tr_flat)
        te_ordered_chunks.append(b_te_flat)

    tr_idx = np.concatenate(tr_ordered_chunks)
    te_idx = np.concatenate(te_ordered_chunks)

    preprocess = T.Compose([
        T.Resize((img_size, img_size), antialias=True),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    print(f"Formatting tensors to {img_size}x{img_size}...")

    def extract_and_format(indices):
        x_sel = imgs[indices]
        x_tensor = torch.from_numpy(x_sel).permute(0, 3, 1, 2).float() / 255.0
        x_tensor = preprocess(x_tensor)

        # NumPy arrays for labels to match MNIST output
        y_array = y_full[indices]
        return x_tensor, y_array

    X_train_all, y_train_all = extract_and_format(tr_idx)
    X_test, y_test = extract_and_format(te_idx)

    return X_train_all, y_train_all, X_test, y_test, tr_idx, te_idx