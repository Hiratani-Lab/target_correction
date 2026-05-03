import argparse
import math
import numpy as np
import matplotlib.pyplot as plt
import torch
import pandas as pd
import os
import sys
import time
import warnings
from pathlib import Path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))
from nn_model import train_core50_van_sgd, train_core50_van_sgd_cumulative, train_core50_van_sgd_ewc, build_resnet18_model
from dataset import load_core50_subset_whitened
warnings.filterwarnings("ignore", message=".*tf.NodeDef is deprecated.*")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

def mean_and_sem(A_2d: np.ndarray):
    A = np.asarray(A_2d, dtype=np.float64)
    mean = A.mean(axis=0)
    if A.shape[0] <= 1:
        sem = np.zeros_like(mean)
    else:
        sem = A.std(axis=0, ddof=1) / math.sqrt(A.shape[0])
    return mean, sem

def kernel_save_path(hy_params, root_dir, seed, ext="npy", prefix="K"):
    dataset, train_order_type = hy_params['ds_type'], hy_params['train_order_type']
    n_train, n_test = hy_params['Ntrain'], hy_params['Ntest']
    kernel_tp = hy_params['kernel_tp']
    root = Path(root_dir)
    out_dir = root / dataset / train_order_type
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = f"{prefix}_{dataset}_{train_order_type}_ntr{n_train}_nte{n_test}_{kernel_tp}_seed{seed:03d}.{ext}"
    return out_dir / fname

def mse_from_f(f_cxm: np.ndarray, Z_mxc: np.ndarray, half: bool = True) -> float:
    """
    f_cxm: (C, M) score matrix
    Z_mxc: (M, C) one-hot (or soft) targets
    returns: mean over samples of sum over classes of squared error
    """
    F = f_cxm.T                       # (M, C)
    diff = F - Z_mxc                  # (M, C)
    se = (diff ** 2).sum(axis=1)      # (M,)  sum over output dim (no averaging over C)
    mse = se.mean()                   # mean over samples
    return 0.5 * mse if half else mse

def step_kernel(s, y_train, y_test, K_train, K_tr_te, hy_params):
    classes_seen = np.unique(y_train[:s])
    max_class_id = np.max(classes_seen)

    t = np.searchsorted(y_test, max_class_id, side='right')

    if hy_params['Class_wise']:
        K_dd = K_train[:, :s, :s]
        K_dq_tr = K_train[:, :s, :s]
        if hy_params['train_order_type'] == 'random':
            K_dq_te = K_tr_te[:, :s, :]
            y_test_sub = y_test
        else:
            K_dq_te = K_tr_te[:, :s, :t]
            y_test_sub = y_test[:t]
    else:
        K_dd = K_train[:s, :s]
        K_dq_tr = K_train[:s, :s]
        if hy_params['train_order_type'] == 'random':
            K_dq_te = K_tr_te[:s, :]
            y_test_sub = y_test
        else:
            K_dq_te = K_tr_te[:s, :t]
            y_test_sub = y_test[:t]
    y_test_sub_oh = np.eye(10)[y_test_sub]
    return K_dd, K_dq_tr, K_dq_te, y_test_sub, y_test_sub_oh

def core50_van_sgd_one_param(sgd_lr=0.001, hy_params_vanilla=None):
    hy_params = hy_params_vanilla.copy()
    hy_params.update({'sgd_lr': sgd_lr})
    num_epochs, batch_size = hy_params['epochs'], hy_params['sgd_batch_size']
    device, kernel_tp, num_seeds, frames_per_session = hy_params['device'], hy_params['kernel_tp'], hy_params['ikmax'], hy_params['frames_per_session']

    # Curves over seeds
    sgd_tr_all, sgd_te_all = [], []
    steps = range(1, 10, 1)
    num_tr_per_task = 8*frames_per_session
    for sd in range(num_seeds):
        print("current seed:", sd)
        model = build_resnet18_model(
            device, num_classes=50, use2dense=hy_params['use2dense'], hidden_dim=100,
            bias_flag=hy_params["bias_dense"], update_cnn=hy_params['update_cnn'],
        )
        X_train_all, y_train_all, X_test_all, y_test_all, tr_idx, te_idx = load_core50_subset_whitened(n_train=None, n_test=None,
                                                                                                       npz_path="core50_imgs.npz",
                                                                                                       pkl_path="paths.pkl",
                                                                                                       frames_per_session=frames_per_session,
                                                                                                       img_size=224,
                                                                                                       seed_select=sd,
                                                                                                       device=device)
        train_accs, test_accs = train_core50_van_sgd(model, X_train_all, y_train_all, X_test_all, y_test_all, hy_params, num_epochs, batch_size, num_tr_per_task, total_classes=50)
        sgd_tr_all.append(train_accs); sgd_te_all.append(test_accs)
    sgd_tr_m, sgd_tr_s = mean_and_sem(np.array(sgd_tr_all))
    sgd_te_m, sgd_te_s = mean_and_sem(np.array(sgd_te_all))

    return steps, sgd_tr_m, sgd_tr_s, sgd_te_m, sgd_te_s


def core50_van_sgd_one_param_cumulative(sgd_lr=0.001, hy_params_vanilla=None):
    hy_params = hy_params_vanilla.copy()
    hy_params.update({'sgd_lr': sgd_lr})
    num_epochs, batch_size = hy_params['epochs'], hy_params['sgd_batch_size']
    device, kernel_tp, num_seeds, frames_per_session = hy_params['device'], hy_params['kernel_tp'], hy_params['ikmax'], hy_params['frames_per_session']

    sgd_tr_all, sgd_te_all = [], []
    steps = range(1, 10, 1)
    num_tr_per_task = 8 * frames_per_session
    for sd in range(num_seeds):
        print("current seed:", sd)
        model = build_resnet18_model(
            device, num_classes=50, use2dense=hy_params['use2dense'], hidden_dim=100,
            bias_flag=hy_params["bias_dense"], update_cnn=hy_params['update_cnn'],
        )
        X_train_all, y_train_all, X_test_all, y_test_all, tr_idx, te_idx = load_core50_subset_whitened(n_train=None, n_test=None,
                                                                                                       npz_path="core50_imgs.npz",
                                                                                                       pkl_path="paths.pkl",
                                                                                                       frames_per_session=frames_per_session,
                                                                                                       img_size=224,
                                                                                                       seed_select=sd,
                                                                                                       device=device)
        train_accs, test_accs = train_core50_van_sgd_cumulative(
            model, X_train_all, y_train_all, X_test_all, y_test_all,
            hy_params, num_epochs, batch_size, num_tr_per_task, total_classes=50
        )
        sgd_tr_all.append(train_accs); sgd_te_all.append(test_accs)
    sgd_tr_m, sgd_tr_s = mean_and_sem(np.array(sgd_tr_all))
    sgd_te_m, sgd_te_s = mean_and_sem(np.array(sgd_te_all))

    return steps, sgd_tr_m, sgd_tr_s, sgd_te_m, sgd_te_s


def core50_van_sgd_one_param_ewc(sgd_lr=0.001, hy_params_vanilla=None):
    hy_params = hy_params_vanilla.copy()
    hy_params.update({'sgd_lr': sgd_lr})
    num_epochs, batch_size = hy_params['epochs'], hy_params['sgd_batch_size']
    device, kernel_tp, num_seeds, frames_per_session = hy_params['device'], hy_params['kernel_tp'], hy_params['ikmax'], hy_params['frames_per_session']

    sgd_tr_all, sgd_te_all = [], []
    steps = range(1, 10, 1)
    num_tr_per_task = 8 * frames_per_session
    for sd in range(num_seeds):
        print("current seed:", sd)
        model = build_resnet18_model(
            device, num_classes=50, use2dense=hy_params['use2dense'], hidden_dim=100,
            bias_flag=hy_params["bias_dense"], update_cnn=hy_params['update_cnn'],
        )
        X_train_all, y_train_all, X_test_all, y_test_all, tr_idx, te_idx = load_core50_subset_whitened(n_train=None, n_test=None,
                                                                                                       npz_path="core50_imgs.npz",
                                                                                                       pkl_path="paths.pkl",
                                                                                                       frames_per_session=frames_per_session,
                                                                                                       img_size=224,
                                                                                                       seed_select=sd,
                                                                                                       device=device)
        train_accs, test_accs = train_core50_van_sgd_ewc(
            model, X_train_all, y_train_all, X_test_all, y_test_all,
            hy_params, num_epochs, batch_size, num_tr_per_task, total_classes=50
        )
        sgd_tr_all.append(train_accs); sgd_te_all.append(test_accs)
    sgd_tr_m, sgd_tr_s = mean_and_sem(np.array(sgd_tr_all))
    sgd_te_m, sgd_te_s = mean_and_sem(np.array(sgd_te_all))

    return steps, sgd_tr_m, sgd_tr_s, sgd_te_m, sgd_te_s


def _fmt_param_for_fname(x: float) -> str:
    s = f"{float(x):.6g}"
    return s.replace("+", "").replace(" ", "_")


def save_van_sgd_to_csv(steps, tr_m, tr_s, te_m, te_s, lr, seeds, cum_replay=True, use_ewc=False, ewc_lambda=None):
    print("learning rate is:", lr)

    num_epochs, batch_size, criterion = hy_params_vanilla['epochs'], hy_params_vanilla['sgd_batch_size'], hy_params_vanilla['criterion']
    num_seeds, frames_per_session, update_cnn = hy_params_vanilla['ikmax'], hy_params_vanilla['frames_per_session'], hy_params_vanilla['update_cnn']
    use2dense, on_cor_bz = hy_params_vanilla['use2dense'], hy_params_vanilla['on_cor_bs']

    if use_ewc:
        lam_part = ""
        if ewc_lambda is not None:
            lam_part = f"_ewcL{_fmt_param_for_fname(ewc_lambda)}"
        results_dir = (
            f"results/van_sgd_ewc_upCNN{update_cnn}_use2dense{use2dense}/epochs{num_epochs}_bz{batch_size}"
            f"_sd{num_seeds}_frame{frames_per_session}_{criterion}_seeds{seeds}{lam_part}"
        )
    elif cum_replay:
        results_dir = f"results/van_sgd_cum_replay_upCNN{update_cnn}_use2dense{use2dense}/epochs{num_epochs}_bz{batch_size}_sd{num_seeds}_frame{frames_per_session}_{criterion}_seeds{seeds}"
    else:
        results_dir = f"results/van_sgd_upCNN{update_cnn}_use2dense{use2dense}/epochs{num_epochs}_bz{batch_size}_sd{num_seeds}_frame{frames_per_session}_{criterion}_seeds{seeds}"
    os.makedirs(results_dir, exist_ok=True)

    lr_str = f"{lr:.6f}".rstrip('0').rstrip('.')

    df = pd.DataFrame({
        'task_step': list(steps),
        'train_mean': tr_m,
        'train_std': tr_s,
        'test_mean': te_m,
        'test_std': te_s
    })

    if use_ewc and ewc_lambda is not None:
        filename = (
            f"core50_van_sgd_{criterion}_lr_{lr_str}_ewcL{_fmt_param_for_fname(ewc_lambda)}_seeds_{num_seeds}_epochs{num_epochs}_bz{batch_size}.csv"
        )
    else:
        filename = f"core50_van_sgd_{criterion}_lr_{lr_str}_seeds_{num_seeds}_epochs{num_epochs}_bz{batch_size}.csv"

    filepath = os.path.join(results_dir, filename)

    df.to_csv(filepath, index=False)

    print(f"Successfully saved results to: {filepath}")

from nn_model import (
    dumb_train_core50_label_correction_adap_kernel,
    efficient_train_core50_label_correction_adap_kernel,
    full_efficient_train_core50_label_correction_adap_kernel,
)

def core50_label_cor_one_param(
    sgd_on_eta=0.001,
    sgd_on_gm=1,
    online_cor_type='batch_gpu',
    hy_params_vanilla=None,
    efficient_label_cor=True,
    label_cor_backend='efficient',
):
    hy_params = hy_params_vanilla.copy()
    print("eta:", sgd_on_eta, "gm:", sgd_on_gm)

    hy_params.update({'sgd_lr': sgd_on_eta})
    device, kernel_tp, num_seeds, frames_per_session = hy_params['device'], hy_params['kernel_tp'], hy_params['ikmax'], hy_params['frames_per_session']

    # Curves over seeds
    sgd_tr_all, sgd_te_all = [], []
    steps = range(1, 10, 1)
    for sd in range(num_seeds):
        print("current seed:", sd)
        model = build_resnet18_model(
            device, num_classes=50, use2dense=hy_params['use2dense'], hidden_dim=100,
            bias_flag=hy_params["bias_dense"], update_cnn=hy_params['update_cnn'],
        )
        X_train_all, y_train_all, X_test_all, y_test_all, tr_idx, te_idx = load_core50_subset_whitened(n_train=None, n_test=None,
                                                                                                       npz_path="core50_imgs.npz",
                                                                                                       pkl_path="paths.pkl",
                                                                                                       frames_per_session=frames_per_session,
                                                                                                       img_size=224,
                                                                                                       seed_select=sd,
                                                                                                       device=device)
        if efficient_label_cor:
            if label_cor_backend == 'full_efficient':
                train_accs, test_accs = full_efficient_train_core50_label_correction_adap_kernel(
                    model, X_train_all, y_train_all, X_test_all, y_test_all, hy_params,
                    sgd_on_eta=sgd_on_eta, sgd_on_gm=sgd_on_gm, online_cor_type=online_cor_type
                )
            else:
                train_accs, test_accs = efficient_train_core50_label_correction_adap_kernel(
                    model, X_train_all, y_train_all, X_test_all, y_test_all, hy_params,
                    sgd_on_eta=sgd_on_eta, sgd_on_gm=sgd_on_gm, online_cor_type=online_cor_type
                )
        else:
            train_accs, test_accs = dumb_train_core50_label_correction_adap_kernel(model, X_train_all, y_train_all, X_test_all, y_test_all, hy_params, sgd_on_eta=sgd_on_eta, sgd_on_gm=sgd_on_gm, online_cor_type=online_cor_type)
        sgd_tr_all.append(train_accs); sgd_te_all.append(test_accs)
    sgd_tr_m, sgd_tr_s = mean_and_sem(np.array(sgd_tr_all))
    sgd_te_m, sgd_te_s = mean_and_sem(np.array(sgd_te_all))

    return steps, sgd_tr_m, sgd_tr_s, sgd_te_m, sgd_te_s

def save_on_cor_to_csv(
    steps,
    tr_m,
    tr_s,
    te_m,
    te_s,
    eta,
    gm,
    online_cor_type,
    efficient_label_cor,
    hy_params_vanilla,
    seeds=10,
    label_cor_backend='efficient',
):
    print("eta:", eta, "gamma:", gm, "online_cor_type:", online_cor_type)

    num_epochs, batch_size, criterion = hy_params_vanilla['epochs'], hy_params_vanilla['sgd_batch_size'], hy_params_vanilla['criterion']
    num_seeds, frames_per_session, update_cnn = hy_params_vanilla['ikmax'], hy_params_vanilla['frames_per_session'], hy_params_vanilla['update_cnn']
    use2dense, on_cor_bz = hy_params_vanilla['use2dense'], hy_params_vanilla['on_cor_bs']

    # Dictionaries to store both Mean (_m) and SEM (_s)
    cor_tag = 'dumb'
    if efficient_label_cor:
        cor_tag = label_cor_backend
    results_dir = (
        f"results/online_correction_{cor_tag}_upCNN{update_cnn}_2dense{use2dense}_z_bz{hy_params_vanilla['z_mini_bz']}"
        f"/epochs{num_epochs}_bz{batch_size}_on_cor_bz{on_cor_bz}_sd{num_seeds}_frame{frames_per_session}_{criterion}_{online_cor_type}_seeds{seeds}"
    )
    os.makedirs(results_dir, exist_ok=True)

    df = pd.DataFrame({
        'task_step': steps,
        'train_mean': tr_m,
        'train_std': tr_s,
        'test_mean': te_m,
        'test_std': te_s
    })

    filename = f"core50_online_correction_{criterion}_eta_{eta}_gm{gm}_seeds_{num_seeds}_epochs{num_epochs}_bz{batch_size}.csv"

    filepath = os.path.join(results_dir, filename)

    df.to_csv(filepath, index=False)

    print(f"Successfully saved results to: {filepath}")

if __name__ == "__main__":
    hy_params_vanilla = {
        'device': torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        # network setting
        'use2dense': True,  # whether add another dense layer after flattening
        'use_resnet': True,
        'update_cnn': False,

        'ds_type': 'CORe50',
        'opt_type': 'sgd',  # 'adam', 'sgd'
        'criterion': 'mse',  # 'mse', 'crossentropy'
        'z_type': 'both',  # 'both'/'online'/'offline', online iterative label correction type: both=online+offline
        'fixed_first_ep_kernel': False, # in multi-epochs, whether only use the first epoch's kernel and fix it for later epochs

        'adap_kernel_times': 9,  # number of times to calculate the kernel, usually the number of tasks in task_incre settings
        'frames_per_session': 15,
        'epochs': 10,  # epochs in task_incremental settings
        'ikmax': 5,  # maximum number of seeds
        'KbU_block_size': 4,  # block-wsie size for online KbU: 16, 32
        'Nb': 16,  # 16 the block size for memory efficiency
        'num_classes': 50,  # one hot dimension

        'gm': 0.01,
        'eta': 1,
        'sgd_lr': 0.005,
        'sgd_momentum': 0,
        'sgd_weight_decay': 0,

        'on_cor_bs': 20,  # online iterative label correction batch size
        'sgd_batch_size': 4,
        'z_mini_bz': 4, # mini batch size for KbU

        'kernel_tp': "jacobian_empirical_no_rescale", # 'feature_kernel'/"jacobian_empirical_no_rescale"/"vjp_jvp_no_rescale", "stax_kernel", "jacobian_empirical_rescale" "jacobian_norm", 'jacobian_classwise_no_rescale'
        'Class_wise': False,  # class-wise kernel calculation
        'Class_avg': True,  # average over class number in kernel calculation

        "bias_conv": False,  # True or False
        "bias_dense": False,  # True or False
        "bias_exist": "none", # "full"(bias_conv&bias_dense=True/float), "conv"(only bias_conv=True/float), "none"(bias_conv&bias_dense=False/None)

        "lambda_loss_gate": 0,
        # EWC
        "ewc_lambda": 0,  # 0 if not using ewc
        "ewc_update_every": 100,  # N_train/10
        "ewc_fisher_n": 100000,  # subsample for fisher matrix(optional)
    }
    Begin = time.time()
    print("device:", hy_params_vanilla['device'])
    parser = argparse.ArgumentParser(description='Parallel Hyperparameter Sweep')
    parser.add_argument('--label_correction', type=str, default='none', choices=['none', 'cumulative', 'sgd_ewc', 'cor'],
                        help='none: vanilla task-only; cumulative: cumulative replay on all data seen so far; sgd_ewc: vanilla + EWC; cor: label correction')
    parser.add_argument('--sgd_eta', type=float, default=0.003, help='Learning rate / Eta (label correction path)')
    parser.add_argument('--sgd_gamma', type=float, default=100, help='Gamma (label correction path)')
    parser.add_argument('--sgd_lr', type=float, default=0.01, help='SGD lr (vanilla / cumulative; or fallback if sgd_lr_fixed omitted)')
    parser.add_argument(
        '--sgd_lr_fixed',
        type=float,
        default=None,
        help='fixed SGD lr for sgd_ewc (recommended with --ewc_lambda sweep)',
    )
    parser.add_argument(
        '--ewc_lambda',
        type=float,
        default=None,
        help='EWC lambda for sgd_ewc (override hy_params_vanilla); omit to use dict default',
    )
    parser.add_argument('--online_cor_type', type=str, default='online_correction')
    parser.add_argument(
        '--label_cor_backend',
        type=str,
        default='efficient',
        choices=['efficient', 'full_efficient'],
        help='efficient: task-wise Z update; full_efficient: kernel+Z updated each epoch (last epoch prev tasks + current epoch)',
    )
    parser.add_argument('--use2dense', type=int, default=None, choices=[0, 1],
                        help='1: extra dense after flatten; 0: single dense')
    parser.add_argument('--update_cnn', type=int, default=None, choices=[0, 1],
                        help='1: train CNN layers; 0: freeze (matches hy_params update_cnn)')
    parser.add_argument('--adap_kernel_times', type=int, default=None)
    parser.add_argument('--frames_per_session', type=int, default=None)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--ikmax', type=int, default=None, help='number of random seeds')
    parser.add_argument('--on_cor_bs', type=int, default=None)
    parser.add_argument('--sgd_batch_size', type=int, default=None)
    parser.add_argument('--z_mini_bz', type=int, default=None)
    parser.add_argument('--criterion', type=str, default=None, choices=['mse', 'crossentropy'],
                        help='loss; omit to keep hy_params_vanilla default')
    parser.add_argument('--z_type', type=str, default=None, choices=['both', 'online', 'offline'],
                        help='label correction Z mode; omit to keep hy_params_vanilla default')

    args = parser.parse_args()

    if args.use2dense is not None:
        hy_params_vanilla['use2dense'] = bool(args.use2dense)
    if args.update_cnn is not None:
        hy_params_vanilla['update_cnn'] = bool(args.update_cnn)
    if args.adap_kernel_times is not None:
        hy_params_vanilla['adap_kernel_times'] = args.adap_kernel_times
    if args.frames_per_session is not None:
        hy_params_vanilla['frames_per_session'] = args.frames_per_session
    if args.epochs is not None:
        hy_params_vanilla['epochs'] = args.epochs
    if args.ikmax is not None:
        hy_params_vanilla['ikmax'] = args.ikmax
    if args.on_cor_bs is not None:
        hy_params_vanilla['on_cor_bs'] = args.on_cor_bs
    if args.sgd_batch_size is not None:
        hy_params_vanilla['sgd_batch_size'] = args.sgd_batch_size
    if args.z_mini_bz is not None:
        hy_params_vanilla['z_mini_bz'] = args.z_mini_bz
    if args.criterion is not None:
        hy_params_vanilla['criterion'] = args.criterion
    if args.z_type is not None:
        hy_params_vanilla['z_type'] = args.z_type

    gammas = [0.01, 0.1, 1, 10, 30, 100, 300]
    sgd_eta_list, sgd_gamma_list = [0.003, 0.01, 0.03], [30, 100, 300]
    gate_frac_list = [0.1, 0.3, 0.5, 0.7, 0.999]
    sgd_lr_list = [0.0005, 0.001, 0.003]  
    if args.label_correction == 'none':
        steps, tr_m, tr_s, te_m, te_s = core50_van_sgd_one_param(sgd_lr=args.sgd_lr, hy_params_vanilla=hy_params_vanilla)
        save_van_sgd_to_csv(steps=steps, tr_m=tr_m, tr_s=tr_s, te_m=te_m, te_s=te_s, lr=args.sgd_lr, seeds=args.ikmax, cum_replay=False)
    elif args.label_correction == 'cumulative':
        steps, tr_m, tr_s, te_m, te_s = core50_van_sgd_one_param_cumulative(sgd_lr=args.sgd_lr, hy_params_vanilla=hy_params_vanilla)
        save_van_sgd_to_csv(steps=steps, tr_m=tr_m, tr_s=tr_s, te_m=te_m, te_s=te_s, lr=args.sgd_lr, seeds=args.ikmax, cum_replay=True)
    elif args.label_correction == 'sgd_ewc':
        hp = hy_params_vanilla.copy()
        if args.ewc_lambda is not None:
            hp['ewc_lambda'] = float(args.ewc_lambda)
        lr_use = args.sgd_lr_fixed if args.sgd_lr_fixed is not None else args.sgd_lr
        print("sgd_ewc: lr=", lr_use, "ewc_lambda=", hp.get('ewc_lambda'))
        steps, tr_m, tr_s, te_m, te_s = core50_van_sgd_one_param_ewc(sgd_lr=lr_use, hy_params_vanilla=hp)
        save_van_sgd_to_csv(
            steps=steps,
            tr_m=tr_m,
            tr_s=tr_s,
            te_m=te_m,
            te_s=te_s,
            lr=lr_use,
            seeds=args.ikmax,
            cum_replay=False,
            use_ewc=True,
            ewc_lambda=hp.get('ewc_lambda'),
        )
    else:
        steps, sgd_tr_m, sgd_tr_s, sgd_te_m, sgd_te_s = core50_label_cor_one_param(
            sgd_on_eta=args.sgd_eta,
            sgd_on_gm=args.sgd_gamma,
            online_cor_type=args.online_cor_type,
            hy_params_vanilla=hy_params_vanilla,
            efficient_label_cor=True,
            label_cor_backend=args.label_cor_backend,
        )
        save_on_cor_to_csv(
            steps,
            sgd_tr_m,
            sgd_tr_s,
            sgd_te_m,
            sgd_te_s,
            args.sgd_eta,
            args.sgd_gamma,
            online_cor_type=args.online_cor_type,
            efficient_label_cor=True,
            hy_params_vanilla=hy_params_vanilla,
            seeds=args.ikmax,
            label_cor_backend=args.label_cor_backend,
        )

    End = time.time()
    print("time cost:", End-Begin)