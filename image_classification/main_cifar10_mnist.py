import argparse
import math
import numpy as np
import matplotlib.pyplot as plt
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))
import torch
from nn_model import one_hot, sgd_train_streaming_gating_ewc, sgd_train_streaming_gating_ewc_cumulative, sgd_train_streaming_MSE_fixed_kernel_f0, compute_corrected_Z_iter, compute_corrected_Z_full, sgd_train_streaming_MSE_on_cor_adap_kernel
from nn_model import krr_predict_precomputed, on_predict_eq1_precomputed, on_predict_classwise, krr_predict_classwise, SmallCNN_model, sgd_train_streaming_gating_ewc_epochs
from nn_model import sgd_train_streaming_MSE_online_correction_epochs, kernel_type
from dataset import make_order_multiclass, load_ds
from plot import save_sgd_online_correction_csv, save_sgd_online_correction_seed_csv, plot_all_sgd_online_correction, save_sgd_vanilla_csv, save_sgd_ewc_csv

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import time
import warnings
warnings.filterwarnings("ignore", message=".*tf.NodeDef is deprecated.*")
from pathlib import Path

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

def run_online_one_param(
    order_type="random",          # "random" or "class_incremental"
    eta=0.1,
    hy_params_vanilla=None,
):
    hy_params=hy_params_vanilla.copy()
    hy_params.update({
        'eta': eta,
    })
    device, eval_every = hy_params['device'], hy_params['record_step']
    kernel_tp, bias_exist = hy_params['kernel_tp'], hy_params['bias_exist']
    n_train, n_test, num_seeds = hy_params['Ntrain'], hy_params['Ntest'], hy_params['ikmax']

    # Steps to evaluate
    steps = list(range(eval_every, n_train + 1, eval_every))

    # Curves over seeds
    online_tr_all, online_te_all = [], []
    mse_tr_iter_all, mse_te_iter_all = [], []

    for sd in range(num_seeds):
        print(f"\n=== Seed {sd} | order={order_type} ===")
        torch.manual_seed(sd)
        np.random.seed(sd)
        # Load dataset
        X_train_all, y_train_all, X_test_all, y_test_all, tr_idx, te_idx = load_ds(n_train, n_test, root="./data", seed_select=sd, device=device, hy_params=hy_params)
        # Order training stream
        order = make_order_multiclass(y_train_all, seed=sd, order_type=order_type, n_classes=10)
        test_order = make_order_multiclass(y_test_all, seed=sd, order_type=order_type, n_classes=10)

        X_train = X_train_all[order]
        y_train = y_train_all[order]
        X_test = X_test_all[test_order]
        y_test = y_test_all[test_order]
        Y_train_oh = one_hot(y_train, 10)

        torch.manual_seed(sd)
        model_ntk = SmallCNN_model(hy_params, use_gate=False)
        model_ntk = model_ntk.to(device)
        K_train, K_tr_te = kernel_type(model_ntk, X_train, X_test, device, kernel_tp, hy_params)

        # save kernel
        root_dir = "./kernels"
        dataset = hy_params_vanilla['ds_type']
        train_order_type = hy_params_vanilla['train_order_type']
        # --- Save full train kernel as .npy/ ---
        tr_full_path = kernel_save_path(hy_params, root_dir, sd, ext="npy", prefix="K_tr")
        np.save(tr_full_path, K_train.astype(np.float32))
        print("Saved full K_tr to:", tr_full_path)
        # --- Save full train-test kernel as .npy/ ---
        tr_te_full_path = kernel_save_path(hy_params, root_dir, sd, ext="npy", prefix="K_tr_te")
        np.save(tr_te_full_path, K_tr_te.astype(np.float32))
        print("Saved full K_tr_te to:", tr_te_full_path)
        # --- Save upper triangle (incl diagonal) as .npz ---
        n = K_train.shape[0]
        iu = np.triu_indices(n)  # includes diagonal
        triu_path = kernel_save_path(hy_params, root_dir, sd, ext="npz", prefix="K_tr_triu")
        np.savez(triu_path, n=n, data=K_train[iu].astype(np.float32))

        # --- Verify saved triu file ---
        tmp = np.load(triu_path)  # loads the .npz
        n2 = int(tmp["n"])
        data2 = tmp["data"]  # 1D array of upper-triangle values
        iu2 = np.triu_indices(n2)  # same indexing scheme (includes diagonal)
        # reconstruct full matrix from triu data
        K_tr_rec = np.empty((n2, n2), dtype=data2.dtype)
        K_tr_rec[iu2] = data2
        K_tr_rec[(iu2[1], iu2[0])] = data2  # mirror to lower triangle
        # compare to original
        print("K_tr_rec shape/dtype:", K_tr_rec.shape, K_tr_rec.dtype)
        print("max abs diff:", np.max(np.abs(K_tr_rec - K_train.astype(np.float32))))
        print("K_train sample original [0:3,0:3]:\n", K_train[:3, :3])
        print("K_train sample recon    [0:3,0:3]:\n", K_tr_rec[:3, :3])
        K_tr_te_loaded = np.load(tr_te_full_path)
        print("max abs diff:", np.max(np.abs(K_tr_te_loaded - K_tr_te)))
        print("K_tr_te sample original:\n", K_tr_te[:3, :3])
        print("K_tr_te sample loaded:\n", K_tr_te_loaded[:3, :3])

        # Accuracy vs steps for kernel methods
        online_tr, online_te = [], []
        mse_tr, mse_te = [], []

        for s in steps:
            y_train_sub_oh = Y_train_oh[:s]

            K_dd, K_dq_tr, K_dq_te, y_test_sub, y_test_sub_oh = step_kernel(s, y_train, y_test, K_train, K_tr_te, hy_params)

            # Online learning with Yd
            if hy_params['Class_wise']:
                f_tr, yhat_tr_online = on_predict_classwise(K_dd, K_dq_tr, y_train_sub_oh, eta=eta, block_size=hy_params['KbU_block_size'])
                f_te, yhat_te_online = on_predict_classwise(K_dd, K_dq_te, y_train_sub_oh, eta=eta, block_size=hy_params['KbU_block_size'])
            else:
                f_tr, yhat_tr_online = on_predict_eq1_precomputed(K_dd, K_dq_tr, y_train_sub_oh, eta=eta, block_size=hy_params['KbU_block_size'])
                f_te, yhat_te_online = on_predict_eq1_precomputed(K_dd, K_dq_te, y_train_sub_oh, eta=eta, block_size=hy_params['KbU_block_size'])
            online_tr.append(np.mean(yhat_tr_online == y_train[:s]))
            online_te.append(np.mean(yhat_te_online == y_test_sub))

            # MSE on prediction scores vs one-hot targets
            mse_tr.append(mse_from_f(f_tr, y_train_sub_oh, half=True))
            mse_te.append(mse_from_f(f_te, y_test_sub_oh, half=True))

        online_tr_all.append(online_tr); online_te_all.append(online_te)
        mse_tr_iter_all.append(mse_tr); mse_te_iter_all.append(mse_te)

    online_tr_m, online_tr_s = mean_and_sem(online_tr_all)
    online_te_m, online_te_s = mean_and_sem(online_te_all)
    mse_tr_m, mse_tr_s = mean_and_sem(mse_tr_iter_all)
    mse_te_m, mse_te_s = mean_and_sem(mse_te_iter_all)

    return steps, online_tr_m, online_tr_s, online_te_m, online_te_s, mse_tr_m, mse_tr_s, mse_te_m, mse_te_s


def run_online_cor_one_param(
    order_type="random",          # "random" or "class_incremental"
    eta=0.1,
    gamma=1,
    hy_params_vanilla=None,
    online_cor_type='iter',  # 'iter'/'full'
):
    hy_params=hy_params_vanilla.copy()
    hy_params.update({
        'gm': gamma,
        'eta': eta,
    })
    device, eval_every = hy_params['device'], hy_params['record_step']
    kernel_tp, bias_exist = hy_params['kernel_tp'], hy_params['bias_exist']
    n_train, n_test, num_seeds = hy_params['Ntrain'], hy_params['Ntest'], hy_params['ikmax']

    # Steps to evaluate
    steps = list(range(eval_every, n_train + 1, eval_every))

    # Curves over seeds
    cor_tr_iter_all, cor_te_iter_all = [], []
    mse_tr_iter_all, mse_te_iter_all = [], []

    for sd in range(num_seeds):
        print(f"\n=== Seed {sd} | order={order_type} ===")
        torch.manual_seed(sd)
        np.random.seed(sd)
        # Load dataset
        X_train_all, y_train_all, X_test_all, y_test_all, tr_idx, te_idx = load_ds(n_train, n_test, root="./data", seed_select=sd, device=device, hy_params=hy_params)

        # Order training stream
        order = make_order_multiclass(y_train_all, seed=sd, order_type=order_type, n_classes=10)
        test_order = make_order_multiclass(y_test_all, seed=sd, order_type=order_type, n_classes=10)

        X_train = X_train_all[order]
        y_train = y_train_all[order]
        Y_train_oh = one_hot(y_train, 10)
        # Order test stream
        X_test = X_test_all[test_order]
        y_test = y_test_all[test_order]
        print("Yoh.shape", Y_train_oh.shape)
        # Use its own freshly initialized model (random features baseline)
        torch.manual_seed(sd)
        model_ntk = SmallCNN_model(hy_params, use_gate=False).to(device)

        K_train, K_tr_te = kernel_type(model_ntk, X_train, X_test, device, kernel_tp, hy_params)

        # Calculation of the corrected label Z
        if online_cor_type == 'iter':
            Z_cor = compute_corrected_Z_iter(K_train, Y_train_oh, gamma=gamma, eta=eta)
        elif online_cor_type == 'full':
            Z_cor = compute_corrected_Z_full(K_train, Y_train_oh, gamma=gamma, eta=eta)
        num_classes = Z_cor.shape[1]

        # Accuracy vs steps for kernel methods
        cor_tr_iter, cor_te_iter = [], []
        mse_tr_iter, mse_te_iter = [], []

        for s in steps:
            K_dd, K_dq_tr, K_dq_te, y_test_sub, y_test_sub_oh = step_kernel(s, y_train, y_test, K_train, K_tr_te, hy_params)
            Zd_cor = Z_cor[:s]

            # Corrected online learning with iteratively corrected Zd
            f_tr_c, yhat_tr_c_iter = on_predict_eq1_precomputed(K_dd, K_dq_tr, Zd_cor, eta=eta, block_size=hy_params['KbU_block_size'])
            f_te_c, yhat_te_c_iter = on_predict_eq1_precomputed(K_dd, K_dq_te, Zd_cor, eta=eta, block_size=hy_params['KbU_block_size'])

            cor_tr_iter.append(np.mean(yhat_tr_c_iter == y_train[:s]))
            cor_te_iter.append(np.mean(yhat_te_c_iter == y_test_sub))

            # MSE on prediction scores vs one-hot targets
            mse_tr_iter.append(mse_from_f(f_tr_c, Zd_cor, half=True))
            mse_te_iter.append(mse_from_f(f_te_c, y_test_sub_oh, half=True))

        cor_tr_iter_all.append(cor_tr_iter); cor_te_iter_all.append(cor_te_iter)
        mse_tr_iter_all.append(mse_tr_iter); mse_te_iter_all.append(mse_te_iter)

    # Aggregate mean +/- SEM
    cor_tr_fixed_m, cor_tr_fixed_s = mean_and_sem(cor_tr_iter_all)
    cor_te_fixed_m, cor_te_fixed_s = mean_and_sem(cor_te_iter_all)

    mse_tr_fixed_m, mse_tr_fixed_s = mean_and_sem(mse_tr_iter_all)
    mse_te_fixed_m, mse_te_fixed_s = mean_and_sem(mse_te_iter_all)

    return steps, cor_tr_fixed_m, cor_tr_fixed_s, cor_te_fixed_m, cor_te_fixed_s, mse_tr_fixed_m, mse_tr_fixed_s, mse_te_fixed_m, mse_te_fixed_s


def run_online_params_sweep_plt(eta_list, gamma_list, hy_params_vanilla, label_correction, online_cor_type, write_csv=True):
    block_size = hy_params_vanilla['KbU_block_size']
    train_order_type = hy_params_vanilla['train_order_type']
    if label_correction:
        save_dir = f"results_{hy_params_vanilla['ds_type']}/online_cor_{online_cor_type}_no_classwise_order_{train_order_type}"
    elif hy_params_vanilla['Class_wise']:
        save_dir = f"results_{hy_params_vanilla['ds_type']}/online_no_correction_classwise_order_{train_order_type}"
    else:
        save_dir = f"results_{hy_params_vanilla['ds_type']}/online_no_correction_no_classwise_order_{train_order_type}"

    if hy_params_vanilla['kernel_tp'] == 'jacobian_empirical_no_rescale':
        if hy_params_vanilla['Class_avg']:
            fig_path1 = os.path.join(save_dir, f"train_curve_jac_no_recale_c_avg_block{block_size}.png")
            fig_path2 = os.path.join(save_dir, f"test_curve_jac_no_recale_c_avg_block{block_size}.png")
            fig_path_mse_tr = os.path.join(save_dir, f"train_mse_jac_no_recale_c_avg_block{block_size}.png")
            fig_path_mse_te = os.path.join(save_dir, f"test_mse_jac_no_recale_c_avg_block{block_size}.png")
        else:
            fig_path1 = os.path.join(save_dir, f"train_curve_jac_no_recale_block{block_size}.png")
            fig_path2 = os.path.join(save_dir, f"test_curve_jac_no_recale_block{block_size}.png")
            fig_path_mse_tr = os.path.join(save_dir, f"train_mse_jac_no_recale_block{block_size}.png")
            fig_path_mse_te = os.path.join(save_dir, f"test_mse_jac_no_recale_block{block_size}.png")

    elif hy_params_vanilla['kernel_tp'] == 'auto_grad':
        fig_path1 = os.path.join(save_dir, f"train_curve_auto_grad_no_scale_block{block_size}.png")
        fig_path2 = os.path.join(save_dir, f"test_curve_auto_grad_no_scale_block{block_size}.png")
    else:
        fig_path1 = os.path.join(save_dir, f"train_curve_jac_recale_block{block_size}.png")
        fig_path2 = os.path.join(save_dir, f"test_curve_jac_recale_block{block_size}.png")
    os.makedirs(save_dir, exist_ok=True)

    # --- Train figure ---
    fig_tr, ax_tr = plt.subplots()
    ax_tr.set_xlabel("prefix length")
    ax_tr.set_ylabel("accuracy")
    ax_tr.grid(True, alpha=0.3)
    # --- Test figure ---
    fig_te, ax_te = plt.subplots()
    ax_te.set_xlabel("prefix length")
    ax_te.set_ylabel("accuracy")
    ax_te.grid(True, alpha=0.3)

    fig_mse_tr, ax_mse_tr = plt.subplots()
    ax_mse_tr.set_xlabel("prefix length")
    ax_mse_tr.set_ylabel("MSE")
    ax_mse_tr.grid(True, alpha=0.3)

    fig_mse_te, ax_mse_te = plt.subplots()
    ax_mse_te.set_xlabel("prefix length")
    ax_mse_te.set_ylabel("MSE")
    ax_mse_te.grid(True, alpha=0.3)

    if label_correction:
        ax_tr.set_title(f"Train accuracy (online + label {online_cor_type} correction)")
        ax_te.set_title(f"Test accuracy (online + label {online_cor_type} correction)")
        ax_mse_tr.set_title(f"Train MSE (online + label {online_cor_type} correction)")
        ax_mse_te.set_title(f"Test MSE (online + label {online_cor_type} correction)")
    else:
        ax_tr.set_title(f"Train accuracy (online vanilla)")
        ax_te.set_title(f"Test accuracy (online vanilla)")
        ax_mse_tr.set_title(f"Train MSE (online vanilla)")
        ax_mse_te.set_title(f"Test MSE (online vanilla)")

    def save_online_reg_csv(
        steps,
        tr_acc_m, tr_acc_s, te_acc_m, te_acc_s,
        tr_mse_m, tr_mse_s, te_mse_m, te_mse_s,
        eta, gamma, save_dir, label_correction, online_cor_type, train_order_type
    ):
        """Save one online kernel-regression run (acc + MSE) to CSV."""
        if label_correction:
            filename = f"online_reg_{online_cor_type}_eta{eta:g}_gamma{gamma:g}.csv"
        else:
            filename = f"online_reg_vanilla_eta{eta:g}.csv"
        save_path = os.path.join(save_dir, filename)
        df = pd.DataFrame({
            'step': steps,
            'train_acc_mean': tr_acc_m,
            'train_acc_sem': tr_acc_s,
            'test_acc_mean': te_acc_m,
            'test_acc_sem': te_acc_s,
            'train_mse_mean': tr_mse_m,
            'train_mse_sem': tr_mse_s,
            'test_mse_mean': te_mse_m,
            'test_mse_sem': te_mse_s,
            'eta': eta,
            'gamma': (gamma if gamma is not None else np.nan),
            'label_correction': label_correction,
            'online_cor_type': online_cor_type,
            'train_order_type': train_order_type,
        })
        df.to_csv(save_path, index=False)
        print(f"Saved online kernel-regression CSV: {save_path}")

    if label_correction:
        for eta in eta_list:
            for gamma in gamma_list:
                print("eta=", eta, "gamma=", gamma)
                steps, cor_tr_fixed_m, cor_tr_fixed_s, cor_te_fixed_m, cor_te_fixed_s, mse_tr_fixed_m, mse_tr_fixed_s, mse_te_fixed_m, mse_te_fixed_s = run_online_cor_one_param(order_type=train_order_type, eta=eta, gamma=gamma, hy_params_vanilla=hy_params_vanilla, online_cor_type=online_cor_type)
                label = f"eta={eta:g}, gamma={gamma:g}"
                print(f"test accuracy:", f"{cor_te_fixed_m[2:3]}")
                if write_csv:
                    save_online_reg_csv(
                        steps,
                        cor_tr_fixed_m, cor_tr_fixed_s, cor_te_fixed_m, cor_te_fixed_s,
                        mse_tr_fixed_m, mse_tr_fixed_s, mse_te_fixed_m, mse_te_fixed_s,
                        eta=eta, gamma=gamma, save_dir=save_dir,
                        label_correction=label_correction, online_cor_type=online_cor_type,
                        train_order_type=train_order_type
                    )

                # train curve
                ax_tr.plot(steps, cor_tr_fixed_m, label=label)
                ax_tr.set_ylim(0, 1)
                ax_tr.fill_between(steps, cor_tr_fixed_m - cor_tr_fixed_s, cor_tr_fixed_m + cor_tr_fixed_s, alpha=0.2)

                # test curve
                ax_te.plot(steps, cor_te_fixed_m, label=label)
                ax_te.set_ylim(0, 1)
                ax_te.fill_between(steps, cor_te_fixed_m - cor_te_fixed_s, cor_te_fixed_m + cor_te_fixed_s, alpha=0.2)

                # train curve MSE
                ax_mse_tr.plot(steps, mse_tr_fixed_m, label=label)
                ax_mse_tr.set_ylim(0, 1)
                ax_mse_tr.fill_between(steps, mse_tr_fixed_m - mse_tr_fixed_s, mse_tr_fixed_m + mse_tr_fixed_s, alpha=0.2)

                # test curve MSE
                ax_mse_te.plot(steps, mse_te_fixed_m, label=label)
                ax_mse_te.set_ylim(0, 1)
                ax_mse_te.fill_between(steps, mse_te_fixed_m - mse_te_fixed_s, mse_te_fixed_m + mse_te_fixed_s, alpha=0.2)
    else:
        for eta in eta_list:
            print("eta=", eta)
            steps, online_tr_m, online_tr_s, online_te_m, online_te_s, mse_tr_m, mse_tr_s, mse_te_m, mse_te_s = run_online_one_param(order_type=train_order_type, eta=eta, hy_params_vanilla=hy_params_vanilla)
            label = f"eta={eta:g}"
            if write_csv:
                save_online_reg_csv(
                    steps,
                    online_tr_m, online_tr_s, online_te_m, online_te_s,
                    mse_tr_m, mse_tr_s, mse_te_m, mse_te_s,
                    eta=eta, gamma=None, save_dir=save_dir,
                    label_correction=label_correction, online_cor_type=online_cor_type,
                    train_order_type=train_order_type
                )

            # train curve
            ax_tr.plot(steps, online_tr_m, label=label)
            ax_tr.set_ylim(0, 1)
            ax_tr.fill_between(steps, online_tr_m - online_tr_s, online_tr_m + online_tr_s, alpha=0.2)

            # test curve
            ax_te.plot(steps, online_te_m, label=label)
            ax_te.set_ylim(0, 1)
            ax_te.fill_between(steps, online_te_m - online_te_s, online_te_m + online_te_s, alpha=0.2)

            # train curve MSE
            ax_mse_tr.plot(steps, mse_tr_m, label=label)
            ax_mse_tr.set_ylim(0, 1)
            ax_mse_tr.fill_between(steps, mse_tr_m - mse_tr_s, mse_tr_m + mse_tr_s, alpha=0.2)

            # test curve MSE
            ax_mse_te.plot(steps, mse_te_m, label=label)
            ax_mse_te.set_ylim(0, 1)
            ax_mse_te.fill_between(steps, mse_te_m - mse_te_s, mse_te_m + mse_te_s, alpha=0.2)

    ax_tr.legend(fontsize=8)
    ax_te.legend(fontsize=8)
    ax_mse_tr.legend(fontsize=8)
    ax_mse_te.legend(fontsize=8)
    fig_tr.tight_layout()
    fig_te.tight_layout()
    fig_mse_tr.tight_layout()
    fig_mse_te.tight_layout()

    fig_tr.savefig(fig_path1, dpi=200, bbox_inches="tight")
    fig_te.savefig(fig_path2, dpi=200, bbox_inches="tight")
    fig_mse_tr.savefig(fig_path_mse_tr, dpi=200, bbox_inches="tight")
    fig_mse_te.savefig(fig_path_mse_te, dpi=200, bbox_inches="tight")

def visualize_label_correction_class_incremental(
    hy_params_vanilla,
    eta: float,
    gamma: float,
    online_cor_type: str = "iter",
    seed: int = 0,
):
    hy_params = hy_params_vanilla.copy()
    hy_params.update({"eta": eta, "gm": gamma})

    # Save to same result family as online corrected runs
    save_dir = f"results_{hy_params['ds_type']}/online_cor_{online_cor_type}_no_classwise_order_class_incremental"
    os.makedirs(save_dir, exist_ok=True)

    device = hy_params["device"]
    n_train, n_test = hy_params["Ntrain"], hy_params["Ntest"]
    kernel_tp = hy_params["kernel_tp"]

    torch.manual_seed(seed)
    np.random.seed(seed)
    X_train_all, y_train_all, X_test_all, y_test_all, tr_idx, te_idx = load_ds(
        n_train, n_test, root="./data", seed_select=seed, device=device, hy_params=hy_params
    )

    order = make_order_multiclass(y_train_all, seed=seed, order_type="class_incremental", n_classes=10)
    X_train = X_train_all[order]
    y_train = y_train_all[order]
    Y_train_oh = one_hot(y_train, 10)

    model_ntk = SmallCNN_model(hy_params, use_gate=False).to(device)
    K_train, _ = kernel_type(model_ntk, X_train, X_test_all, device, kernel_tp, hy_params)
    if K_train.ndim != 2:
        raise ValueError("Label-correction visualization requires 2D kernel matrix (set Class_wise=False).")

    if online_cor_type == "iter":
        Z_cor = compute_corrected_Z_iter(K_train, Y_train_oh, gamma=gamma, eta=eta)
    elif online_cor_type == "full":
        Z_cor = compute_corrected_Z_full(K_train, Y_train_oh, gamma=gamma, eta=eta)
    else:
        raise ValueError(f"Unsupported online_cor_type={online_cor_type!r}. Use 'iter' or 'full'.")

    Delta = Z_cor - Y_train_oh

    # 1) Pick representative sample per class for three modes: first / middle / last
    rep_z_by_mode = {"first": [], "middle": [], "last": []}
    rep_delta_by_mode = {"first": [], "middle": [], "last": []}
    for c in range(10):
        idx_c = np.where(y_train == c)[0]
        if len(idx_c) == 0:
            raise RuntimeError(f"No sample found for class {c} in the training stream.")
        idx_first = int(idx_c[0])
        idx_middle = int(idx_c[int(len(idx_c) / 2)])  # even n -> floor(n/2), as requested
        idx_last = int(idx_c[-1])
        for mode, idx_sel in (("first", idx_first), ("middle", idx_middle), ("last", idx_last)):
            rep_z_by_mode[mode].append(Z_cor[idx_sel])
            rep_delta_by_mode[mode].append(Delta[idx_sel])
    for mode in ("first", "middle", "last"):
        rep_z_by_mode[mode] = np.asarray(rep_z_by_mode[mode], dtype=np.float64)          # (10,10)
        rep_delta_by_mode[mode] = np.asarray(rep_delta_by_mode[mode], dtype=np.float64)  # (10,10)

    # 2) Class-wise average ||Delta_y||_1
    delta_l1_per_sample = np.abs(Delta).sum(axis=1)  # (N,)
    delta_class_mean = []
    delta_class_sem = []
    for c in range(10):
        vals = delta_l1_per_sample[y_train == c]
        if len(vals) == 0:
            delta_class_mean.append(np.nan)
            delta_class_sem.append(np.nan)
        else:
            delta_class_mean.append(float(np.mean(vals)))
            if len(vals) <= 1:
                delta_class_sem.append(0.0)
            else:
                delta_class_sem.append(float(np.std(vals, ddof=1) / np.sqrt(len(vals))))
    delta_class_mean = np.asarray(delta_class_mean, dtype=np.float64)
    delta_class_sem = np.asarray(delta_class_sem, dtype=np.float64)

    # Optional CSV dump for reproducibility
    # Keep original "first" filenames for backward compatibility, and add explicit mode files.
    pd.DataFrame(rep_z_by_mode["first"], index=np.arange(10), columns=[f"z_{j}" for j in range(10)]).to_csv(
        os.path.join(save_dir, f"label_correction_heatmap_matrix_eta{eta:g}_gm{gamma:g}_seed{seed}.csv")
    )
    pd.DataFrame(rep_delta_by_mode["first"], index=np.arange(10), columns=[f"delta_{j}" for j in range(10)]).to_csv(
        os.path.join(save_dir, f"label_delta_heatmap_matrix_eta{eta:g}_gm{gamma:g}_seed{seed}.csv")
    )
    for mode in ("first", "middle", "last"):
        pd.DataFrame(rep_z_by_mode[mode], index=np.arange(10), columns=[f"z_{j}" for j in range(10)]).to_csv(
            os.path.join(save_dir, f"label_correction_heatmap_matrix_{mode}_eta{eta:g}_gm{gamma:g}_seed{seed}.csv")
        )
        pd.DataFrame(rep_delta_by_mode[mode], index=np.arange(10), columns=[f"delta_{j}" for j in range(10)]).to_csv(
            os.path.join(save_dir, f"label_delta_heatmap_matrix_{mode}_eta{eta:g}_gm{gamma:g}_seed{seed}.csv")
        )
    pd.DataFrame({
        "class_id": np.arange(10),
        "delta_l1_mean": delta_class_mean,
        "delta_l1_sem": delta_class_sem,
    }).to_csv(
        os.path.join(save_dir, f"delta_y_class_curve_eta{eta:g}_gm{gamma:g}_seed{seed}.csv"),
        index=False
    )

    # ---- Plot A/A2: heatmaps for first/middle/last representative samples ----
    for mode in ("first", "middle", "last"):
        rep_mat = rep_z_by_mode[mode]
        rep_delta_mat = rep_delta_by_mode[mode]

        fig_hm, ax_hm = plt.subplots(figsize=(6.4, 5.6))
        im = ax_hm.imshow(rep_mat, cmap="viridis", aspect="equal", vmin=np.min(rep_mat), vmax=np.max(rep_mat))
        ax_hm.set_title(rf"Corrected label $z$ ({mode} sample per class)")
        ax_hm.set_xlabel("Label dimension")
        ax_hm.set_ylabel("Source class ID")
        ax_hm.set_xticks(np.arange(10))
        ax_hm.set_yticks(np.arange(10))
        cbar = fig_hm.colorbar(im, ax=ax_hm, fraction=0.046, pad=0.04)
        cbar.set_label("z value")
        fig_hm.tight_layout()
        hm_png = os.path.join(save_dir, f"label_correction_heatmap_{mode}_eta{eta:g}_gm{gamma:g}_seed{seed}.png")
        fig_hm.savefig(hm_png, dpi=220, bbox_inches="tight")
        fig_hm.savefig(hm_png.replace(".png", ".pdf"), bbox_inches="tight")
        plt.close(fig_hm)

        fig_dhm, ax_dhm = plt.subplots(figsize=(6.4, 5.6))
        vmax = float(np.max(np.abs(rep_delta_mat)))
        vmax = vmax if vmax > 0 else 1e-8
        # Put class ID on x-axis and label dimension on y-axis
        im_d = ax_dhm.imshow(rep_delta_mat.T, cmap="coolwarm", aspect="equal", vmin=-vmax, vmax=vmax)
        ax_dhm.set_title(rf"Label correction $\Delta = z - y$ ({mode} sample per class)")
        ax_dhm.set_xlabel("Class ID")
        ax_dhm.set_ylabel("Label dimension")
        ax_dhm.set_xticks(np.arange(10))
        ax_dhm.set_yticks(np.arange(10))
        cbar_d = fig_dhm.colorbar(im_d, ax=ax_dhm, fraction=0.046, pad=0.04)
        cbar_d.set_label(r"$\Delta$ value")
        fig_dhm.tight_layout()
        dhm_png = os.path.join(save_dir, f"label_delta_heatmap_{mode}_eta{eta:g}_gm{gamma:g}_seed{seed}.png")
        fig_dhm.savefig(dhm_png, dpi=220, bbox_inches="tight")
        fig_dhm.savefig(dhm_png.replace(".png", ".pdf"), bbox_inches="tight")
        plt.close(fig_dhm)

    # ---- Plot B: ||Delta_y||_class ----
    cls = np.arange(10)
    fig_dy, ax_dy = plt.subplots(figsize=(6.6, 4.6))
    lo = np.clip(delta_class_mean - delta_class_sem, 0.0, None)
    hi = delta_class_mean + delta_class_sem
    ax_dy.fill_between(cls, lo, hi, alpha=0.2, color="C1")
    ax_dy.plot(cls, delta_class_mean, "-o", color="C1", linewidth=2.0, markersize=6.0)
    ax_dy.set_title(r"Class-wise correction magnitude: $|z-y|$ (mean $\pm$ SEM)")
    ax_dy.set_xlabel("Class ID")
    ax_dy.set_ylabel(r"$|z-y|$ (class average)")
    ax_dy.set_xticks(np.arange(10))
    ax_dy.grid(True, alpha=0.3, axis="y")
    fig_dy.tight_layout()
    dy_png = os.path.join(save_dir, f"delta_y_class_curve_eta{eta:g}_gm{gamma:g}_seed{seed}.png")
    fig_dy.savefig(dy_png, dpi=220, bbox_inches="tight")
    fig_dy.savefig(dy_png.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig_dy)

    print(f"Saved label-correction visualizations to: {save_dir}")

from dataset import format_split_10class_default_tasks, format_split_binary_tasks
from nn_model import sgd_train_streaming_task_incre_epochs, sgd_train_task_incre_label_correction_epochs_adap_kernel
from plot import save_correction_one_seed_csv, save_sgd_vanilla_one_seed_csv
def run_sgd_one_param(sgd_eta=0.1, sgd_gamma=0.1, sgd_lr=0.001, hy_params_vanilla=None,
    label_correction_type='none', # 'fixed_kernel_f0', 'online_correction', 'none'
    order_type="random", gate_frac=1,  # gate fraction
    gate_mode="none",  # "infer", "none"
    ewc_lambda=0, online_cor_type='iter', save_dir='results'):
    hy_params = hy_params_vanilla.copy()
    hy_params.update({
        'gm': sgd_gamma,
        'eta': sgd_eta,
        'sgd_lr': sgd_lr,
        'gate_frac': gate_frac,
        'gate_mode': gate_mode,
        'ewc_lambda': ewc_lambda
    })
    eval_every, device = hy_params['record_step'], hy_params['device']
    kernel_tp = hy_params['kernel_tp']
    n_train, n_test, num_seeds = hy_params['Ntrain'], hy_params['Ntest'], hy_params['ikmax']
    # Curves over seeds
    sgd_tr_cor_all, sgd_te_cor_all = [], []
    sgd_tr_mse_cor_all, sgd_te_mse_cor_all = [], []
    X_train_all, y_train_all, X_test_all, y_test_all, tr_idx, te_idx = load_ds(n_train, n_test, root="./data", seed_select=42,
                                                                       device=device, hy_params=hy_params)

    for sd in range(num_seeds):
        torch.manual_seed(sd)
        np.random.seed(sd)
        # Load dataset

        # Order training stream
        if hy_params_vanilla['train_order_type'] == 'task_incremental':
            X_train, y_train, X_test, y_test, task_pairs = format_split_binary_tasks(X_train_all, y_train_all, X_test_all, y_test_all, seed=sd)
            Y_train_oh = one_hot(y_train, 2)
        else:
            order = make_order_multiclass(y_train_all, seed=sd, order_type=order_type, n_classes=10)
            X_train = X_train_all[order]
            y_train = y_train_all[order]
            Y_train_oh = one_hot(y_train,10)
            X_test, y_test = X_test_all, y_test_all

        # ----- (A) SGD baseline with iteratively corrected labels -----
        Niters = range(eval_every, n_train + 1, eval_every)
        if gate_mode=='none':
            print("not using gate")
            use_gate=False
        else:
            print("using gate")
            use_gate=True
        model_sgd_cor = SmallCNN_model(hy_params, use_gate).to(device)
        p = next(model_sgd_cor.parameters())
        if label_correction_type == 'fixed_kernel_f0':
            print("using f0 and fixed kernel")
            K_train, K_tr_te = kernel_type(model_sgd_cor, X_train, X_test, device, kernel_tp, hy_params)
            sgd_tr_cor_curve, sgd_te_cor_curve, sgd_tr_cor_mse, sgd_te_cor_mse = sgd_train_streaming_MSE_fixed_kernel_f0(model_sgd_cor, model_sgd_cor, K_train, X_train, y_train, X_test, y_test, Niters, hy_params, label_corrected=True)
            steps = Niters
        elif label_correction_type == 'online_correction':
            if hy_params['epochs']>1:
                steps, sgd_tr_cor_curve, sgd_te_cor_curve, sgd_tr_cor_mse, sgd_te_cor_mse = sgd_train_task_incre_label_correction_epochs_adap_kernel(model_sgd_cor, X_train, y_train, X_test, y_test,
                                                                                                                                                     hy_params, gate_mode=gate_mode, num_epochs=hy_params['epochs'],
                                                                                                                                                     sgd_on_gm=sgd_gamma, sgd_on_eta=sgd_eta, online_cor_type=online_cor_type)
            else:
                sgd_tr_cor_curve, sgd_te_cor_curve, sgd_tr_cor_mse, sgd_te_cor_mse = sgd_train_streaming_MSE_on_cor_adap_kernel(model_sgd_cor, Y_train_oh, X_train,
                                                                                                                                y_train, X_test, y_test, Niters, hy_params,
                                                                                                                                sgd_on_gm=sgd_gamma, sgd_on_eta=sgd_eta, online_cor_type=online_cor_type)
                steps = Niters
            save_correction_one_seed_csv(steps, sgd_tr_cor_curve, sgd_te_cor_curve, sgd_eta, sgd_gamma, label_correction_type, sd, save_dir)
        elif label_correction_type == 'cumulative':
            if hy_params['train_order_type'] != 'task_incremental':
                raise ValueError("cumulative only supports train_order_type=='task_incremental'")
            if hy_params['epochs'] != 1:
                raise ValueError("cumulative only supports epochs==1 (use run_sgd_params_sweep_plt, not multi-epoch epochs path)")
            sgd_tr_cor_curve, sgd_te_cor_curve, sgd_tr_cor_mse, sgd_te_cor_mse = sgd_train_streaming_gating_ewc_cumulative(
                model_sgd_cor, X_train, y_train, X_test, y_test, Niters, hy_params, gate_mode)
            steps = Niters
            save_sgd_cumulative_one_seed_csv(steps, sgd_tr_cor_curve, sgd_te_cor_curve, sgd_lr, sd, save_dir)
        else:
            if hy_params['epochs'] > 1:
                steps, sgd_tr_cor_curve, sgd_te_cor_curve, sgd_tr_cor_mse, sgd_te_cor_mse = sgd_train_streaming_task_incre_epochs(model_sgd_cor, X_train, y_train, X_test, y_test, hy_params, gate_mode, num_epochs=hy_params['epochs'])
            else:
                sgd_tr_cor_curve, sgd_te_cor_curve, sgd_tr_cor_mse, sgd_te_cor_mse = sgd_train_streaming_gating_ewc(model_sgd_cor, X_train, y_train, X_test, y_test, Niters, hy_params, gate_mode)
                steps = Niters
            save_sgd_vanilla_one_seed_csv(steps, sgd_tr_cor_curve, sgd_te_cor_curve, sgd_lr, sd, save_dir)
        sgd_tr_cor_all.append(sgd_tr_cor_curve)
        sgd_te_cor_all.append(sgd_te_cor_curve)
        sgd_tr_mse_cor_all.append(sgd_tr_cor_mse); sgd_te_mse_cor_all.append(sgd_te_cor_mse)
    # Aggregate mean +/- SEM
    sgd_tr_cor_m, sgd_tr_cor_s = mean_and_sem(sgd_tr_cor_all)
    sgd_te_cor_m, sgd_te_cor_s = mean_and_sem(sgd_te_cor_all)
    sgd_tr_mse_m, sgd_tr_mse_s = mean_and_sem(sgd_tr_mse_cor_all)
    sgd_te_mse_m, sgd_te_mse_s = mean_and_sem(sgd_te_mse_cor_all)

    return steps, sgd_tr_cor_m, sgd_tr_cor_s, sgd_te_cor_m, sgd_te_cor_s, sgd_tr_mse_m, sgd_tr_mse_s, sgd_te_mse_m, sgd_te_mse_s

from plot import save_classification_csv, save_classification_vanilla_csv, save_classification_cumulative_csv, save_classification_ewc_csv, save_classification_gating_csv
from plot import save_sgd_cumulative_one_seed_csv
def run_sgd_params_sweep_plt(sgd_eta, sgd_gamma, sgd_lr_list, sgd_lr_fixed, hy_params_vanilla, label_correction_type='none', gate_frac_list=None, gate_mode='infer', ewc_lambda_list=None, online_cor_type='batch_gpu'):
    # label_correction_type: 'fixed_kernel_f0', 'online_correction', 'none', 'cumulative', ...
    if gate_frac_list is None:
        gate_frac_list = [1]
    adap_kernel_times = hy_params_vanilla['adap_kernel_times']
    kernel_tp, mini_bz = hy_params_vanilla['kernel_tp'], hy_params_vanilla['sgd_batch_size']
    train_order_type, z_mini_bz = hy_params_vanilla['train_order_type'], hy_params_vanilla['z_mini_bz']
    seeds, epochs = hy_params_vanilla['ikmax'], hy_params_vanilla['epochs']
    on_cor_bs, z_type, first_ep_kernel = hy_params_vanilla['on_cor_bs'], hy_params_vanilla['z_type'], hy_params_vanilla['fixed_first_ep_kernel']
    Ntrain = hy_params_vanilla['Ntrain']
    if label_correction_type not in ('none', 'cumulative'):
        save_dir = f"results_{hy_params_vanilla['ds_type']}_{hy_params_vanilla['criterion']}_Ntr{Ntrain}_seeds{seeds}_sgd_bz{mini_bz}_epochs{epochs}/sgd_correction_cor_tp_{online_cor_type}_kernel_adapN{adap_kernel_times}/{label_correction_type}_order_{train_order_type}_epoch{hy_params_vanilla['epochs']}_{hy_params_vanilla['opt_type']}_mse_on_bs{on_cor_bs}_z_bz{z_mini_bz}_z{z_type}_fixed_first{first_ep_kernel}"
        fig_path1 = os.path.join(save_dir, f"train_curve_sgd_lr{sgd_lr_fixed}_{kernel_tp}_record{hy_params_vanilla['record_step']}.png")
        fig_path2 = os.path.join(save_dir, f"test_curve_sgd_lr{sgd_lr_fixed}_{kernel_tp}_record{hy_params_vanilla['record_step']}.png")
        fig_path_mse_tr = os.path.join(save_dir, f"train_mse_sgd_lr{sgd_lr_fixed}_{kernel_tp}_record{hy_params_vanilla['record_step']}.png")
        fig_path_mse_te = os.path.join(save_dir, f"test_mse_sgd_lr{sgd_lr_fixed}_{kernel_tp}_record{hy_params_vanilla['record_step']}.png")
    elif ewc_lambda_list != None:
        save_dir = f"results_{hy_params_vanilla['ds_type']}_{hy_params_vanilla['criterion']}_Ntr{Ntrain}_seeds{seeds}_sgd_bz{mini_bz}_epochs{epochs}/sgd_ewc_{hy_params_vanilla['criterion']}/cor_{label_correction_type}_order_{train_order_type}_epoch{hy_params_vanilla['epochs']}_{hy_params_vanilla['opt_type']}_lr{sgd_lr_fixed}"
        fig_path1 = os.path.join(save_dir, f"train_acc_lr{sgd_lr_fixed}_{kernel_tp}_record{hy_params_vanilla['record_step']}.png")
        fig_path2 = os.path.join(save_dir, f"test_acc_lr{sgd_lr_fixed}_{kernel_tp}_record{hy_params_vanilla['record_step']}.png")
        fig_path_mse_tr = os.path.join(save_dir, f"train_mse_sgd_lr{sgd_lr_fixed}_{kernel_tp}_record{hy_params_vanilla['record_step']}.png")
        fig_path_mse_te = os.path.join(save_dir, f"test_mse_sgd_lr{sgd_lr_fixed}_{kernel_tp}_record{hy_params_vanilla['record_step']}.png")
    elif label_correction_type == 'cumulative':
        save_dir = f"results_{hy_params_vanilla['ds_type']}_{hy_params_vanilla['criterion']}_Ntr{Ntrain}_seeds{seeds}_sgd_bz{mini_bz}_epochs{epochs}/sgd_{gate_mode}_{hy_params_vanilla['criterion']}/cumulative_order_{train_order_type}_epoch{hy_params_vanilla['epochs']}_{hy_params_vanilla['opt_type']}_gate_{gate_mode}"
        fig_path1 = os.path.join(save_dir, f"train_curve_batch{hy_params_vanilla['sgd_batch_size']}_{kernel_tp}_record{hy_params_vanilla['record_step']}.png")
        fig_path2 = os.path.join(save_dir, f"test_curve_batch{hy_params_vanilla['sgd_batch_size']}_{kernel_tp}_record{hy_params_vanilla['record_step']}.png")
        fig_path_mse_tr = os.path.join(save_dir, f"train_mse_batch{hy_params_vanilla['sgd_batch_size']}_{kernel_tp}_record{hy_params_vanilla['record_step']}.png")
        fig_path_mse_te = os.path.join(save_dir, f"test_mse_batch{hy_params_vanilla['sgd_batch_size']}_{kernel_tp}_record{hy_params_vanilla['record_step']}.png")
    else:
        save_dir = f"results_{hy_params_vanilla['ds_type']}_{hy_params_vanilla['criterion']}_Ntr{Ntrain}_seeds{seeds}_sgd_bz{mini_bz}_epochs{epochs}/sgd_{gate_mode}_{hy_params_vanilla['criterion']}/vanilla_order_{train_order_type}_epoch{hy_params_vanilla['epochs']}_{hy_params_vanilla['opt_type']}_gate_{gate_mode}"
        fig_path1 = os.path.join(save_dir, f"train_curve_batch{hy_params_vanilla['sgd_batch_size']}_{kernel_tp}_record{hy_params_vanilla['record_step']}.png")
        fig_path2 = os.path.join(save_dir, f"test_curve_batch{hy_params_vanilla['sgd_batch_size']}_{kernel_tp}_record{hy_params_vanilla['record_step']}.png")
        fig_path_mse_tr = os.path.join(save_dir, f"train_mse_batch{hy_params_vanilla['sgd_batch_size']}_{kernel_tp}_record{hy_params_vanilla['record_step']}.png")
        fig_path_mse_te = os.path.join(save_dir, f"test_mse_batch{hy_params_vanilla['sgd_batch_size']}_{kernel_tp}_record{hy_params_vanilla['record_step']}.png")
    os.makedirs(save_dir, exist_ok=True)
    print("save_dir exists?", os.path.exists(save_dir), save_dir)
    print("abs save_dir len:", len(os.path.abspath(save_dir)))

    print("fig_path1:", fig_path1)
    print("abs fig_path1 len:", len(os.path.abspath(fig_path1)))
    # --- Train figure ---
    fig_tr, ax_tr = plt.subplots()
    ax_tr.set_xlabel("prefix length")
    ax_tr.set_ylabel("accuracy")
    ax_tr.grid(True, alpha=0.3)
    # --- Test figure ---
    fig_te, ax_te = plt.subplots()
    ax_te.set_xlabel("prefix length")
    ax_te.set_ylabel("accuracy")
    ax_te.grid(True, alpha=0.3)
    # Train MSE
    fig_tr_mse, ax_tr_mse = plt.subplots()
    ax_tr_mse.set_xlabel("prefix length")
    ax_tr_mse.set_ylabel("MSE")
    ax_tr_mse.grid(True, alpha=0.3)
    # Test MSE
    fig_te_mse, ax_te_mse = plt.subplots()
    ax_te_mse.set_xlabel("prefix length")
    ax_te_mse.set_ylabel("MSE")
    ax_te_mse.grid(True, alpha=0.3)

    if label_correction_type not in ('none', 'cumulative'):
        ax_tr.set_title(f"Train acc, SGD + correction={label_correction_type}, Gate={gate_mode}, Criterion={hy_params_vanilla['criterion']}")
        ax_te.set_title(f"Test acc, SGD + correction={label_correction_type}, Gate={gate_mode}, Gate={gate_mode}, Criterion={hy_params_vanilla['criterion']}")
        ax_tr_mse.set_title(f"Train MSE, SGD + correction={label_correction_type}, Gate={gate_mode}, Gate={gate_mode}, Criterion={hy_params_vanilla['criterion']}")
        ax_te_mse.set_title(f"Test MSE, SGD + correction={label_correction_type}, Gate={gate_mode}, Gate={gate_mode}, Criterion={hy_params_vanilla['criterion']}")
    elif ewc_lambda_list != None:
        ax_tr.set_title(f"Train acc, SGD+EWC, SGD_fixed_lr={sgd_lr_fixed}, Criterion={hy_params_vanilla['criterion']}")
        ax_te.set_title(f"Test acc, SGD+EWC, SGD_fixed_lr={sgd_lr_fixed}, Criterion={hy_params_vanilla['criterion']}")
        ax_tr_mse.set_title(f"Train MSE, SGD+EWC, SGD_fixed_lr={sgd_lr_fixed}, Criterion={hy_params_vanilla['criterion']}")
        ax_te_mse.set_title(f"Test MSE, SGD+EWC, SGD_fixed_lr={sgd_lr_fixed}, Criterion={hy_params_vanilla['criterion']}")
    elif label_correction_type == 'cumulative':
        ax_tr.set_title(f"Train acc, SGD + cumulative replay, Gate={gate_mode}, Criterion={hy_params_vanilla['criterion']}")
        ax_te.set_title(f"Test acc, SGD + cumulative replay, Gate={gate_mode}, Criterion={hy_params_vanilla['criterion']}")
        ax_tr_mse.set_title(f"Train MSE, SGD + cumulative replay, Gate={gate_mode}, Criterion={hy_params_vanilla['criterion']}")
        ax_te_mse.set_title(f"Test MSE, SGD + cumulative replay, Gate={gate_mode}, Criterion={hy_params_vanilla['criterion']}")
    else:
        ax_tr.set_title(f"Train acc, SGD, Gate={gate_mode}, Criterion={hy_params_vanilla['criterion']}")
        ax_te.set_title(f"Test acc, SGD, Gate={gate_mode}, Criterion={hy_params_vanilla['criterion']}")
        ax_tr_mse.set_title(f"Train MSE, SGD, Gate={gate_mode}, Criterion={hy_params_vanilla['criterion']}")
        ax_te_mse.set_title(f"Test MSE, SGD, Gate={gate_mode}, Criterion={hy_params_vanilla['criterion']}")

    if label_correction_type not in ('none', 'cumulative'):
        print("using label correction")
        print(f"parameters: eta={sgd_eta}, gm={sgd_gamma}")
        steps, sgd_tr_cor_m, sgd_tr_cor_s, sgd_te_cor_m, sgd_te_cor_s, sgd_tr_mse_m, sgd_tr_mse_s, sgd_te_mse_m, sgd_te_mse_s = run_sgd_one_param(sgd_eta=sgd_eta, sgd_gamma=sgd_gamma, sgd_lr=sgd_eta, hy_params_vanilla=hy_params_vanilla, label_correction_type=label_correction_type, order_type=hy_params_vanilla['train_order_type'], online_cor_type=online_cor_type, save_dir=save_dir)
        label = f"lr={sgd_eta:g}, eta={sgd_eta:g}, gamma={sgd_gamma:g}"

        # train curve
        ax_tr.plot(steps, sgd_tr_cor_m, label=label)
        ax_tr.set_ylim(0, 1)
        ax_tr.fill_between(steps, sgd_tr_cor_m - sgd_tr_cor_s, sgd_tr_cor_m + sgd_tr_cor_s, alpha=0.2)

        # test curve
        ax_te.plot(steps, sgd_te_cor_m, label=label)
        ax_te.set_ylim(0, 1)
        ax_te.fill_between(steps, sgd_te_cor_m - sgd_te_cor_s, sgd_te_cor_m + sgd_te_cor_s, alpha=0.2)

        # Train MSE
        ax_tr_mse.plot(steps, sgd_tr_mse_m, label=label)
        ax_tr_mse.set_ylim(0, 1)
        ax_tr_mse.fill_between(steps, sgd_tr_mse_m - sgd_tr_mse_s, sgd_tr_mse_m + sgd_tr_mse_s, alpha=0.2)
        # test MSE
        ax_te_mse.plot(steps, sgd_te_mse_m, label=label)
        ax_te_mse.set_ylim(0, 1)
        ax_te_mse.fill_between(steps, sgd_te_mse_m - sgd_te_mse_s, sgd_te_mse_m + sgd_te_mse_s, alpha=0.2)

        save_classification_csv(steps, sgd_tr_cor_m, sgd_tr_cor_s, sgd_te_cor_m, sgd_te_cor_s, sgd_eta,
                                sgd_gamma, label_correction_type, hy_params_vanilla, save_dir)
    elif label_correction_type == 'cumulative':
        hy_params_vanilla.update({'ewc_lambda': 0})
        for sgd_lr in sgd_lr_list:
            for gate_frac in gate_frac_list:
                print(f"cumulative parameters: lr={sgd_lr}, gate_frac={gate_frac}")
                steps, sgd_tr_cor_m, sgd_tr_cor_s, sgd_te_cor_m, sgd_te_cor_s, sgd_tr_mse_m, sgd_tr_mse_s, sgd_te_mse_m, sgd_te_mse_s = run_sgd_one_param(
                    sgd_eta=1, sgd_gamma=1, sgd_lr=sgd_lr,
                    hy_params_vanilla=hy_params_vanilla,
                    label_correction_type='cumulative',
                    order_type=hy_params_vanilla['train_order_type'], gate_frac=gate_frac, gate_mode=gate_mode, save_dir=save_dir)
                if gate_frac != 1:
                    save_classification_gating_csv(steps, sgd_tr_cor_m, sgd_tr_cor_s, sgd_te_cor_m, sgd_te_cor_s, gate_frac,
                                                     sgd_lr, hy_params_vanilla, gate_mode, save_dir)
                else:
                    save_classification_cumulative_csv(steps, sgd_tr_cor_m, sgd_tr_cor_s, sgd_te_cor_m, sgd_te_cor_s, sgd_lr,
                                                       hy_params_vanilla, epochs=epochs, mini_bz=z_mini_bz, save_dir=save_dir)
                if gate_mode == 'none':
                    label = f"sgd_lr={sgd_lr:g} (cumulative)"
                else:
                    label = f"sgd_lr={sgd_lr:g}, gate_frac={gate_frac} (cumulative)"
                ax_tr.plot(steps, sgd_tr_cor_m, label=label)
                ax_tr.set_ylim(0, 1)
                ax_tr.fill_between(steps, sgd_tr_cor_m - sgd_tr_cor_s, sgd_tr_cor_m + sgd_tr_cor_s, alpha=0.2)
                ax_te.plot(steps, sgd_te_cor_m, label=label)
                ax_te.set_ylim(0, 1)
                ax_te.fill_between(steps, sgd_te_cor_m - sgd_te_cor_s, sgd_te_cor_m + sgd_te_cor_s, alpha=0.2)
                ax_tr_mse.plot(steps, sgd_tr_mse_m, label=label)
                ax_tr_mse.set_ylim(0, 1)
                ax_tr_mse.fill_between(steps, sgd_tr_mse_m - sgd_tr_mse_s, sgd_tr_mse_m + sgd_tr_mse_s, alpha=0.2)
                ax_te_mse.plot(steps, sgd_te_mse_m, label=label)
                ax_te_mse.set_ylim(0, 1)
                ax_te_mse.fill_between(steps, sgd_te_mse_m - sgd_te_mse_s, sgd_te_mse_m + sgd_te_mse_s, alpha=0.2)
    elif ewc_lambda_list != None:
        print(f"using fixed sgd_lr_fixed={sgd_lr_fixed}")
        for ewc_lambda in ewc_lambda_list:
            print("ewc_lambda=", ewc_lambda)
            steps, sgd_tr_cor_m, sgd_tr_cor_s, sgd_te_cor_m, sgd_te_cor_s, sgd_tr_mse_m, sgd_tr_mse_s, sgd_te_mse_m, sgd_te_mse_s = run_sgd_one_param(sgd_eta=1, sgd_gamma=1, sgd_lr=sgd_lr_fixed,
                                                                                                                                                      hy_params_vanilla=hy_params_vanilla, label_correction_type=label_correction_type,
                                                                                                                                                      order_type=hy_params_vanilla['train_order_type'], gate_frac=1, gate_mode="none", ewc_lambda=ewc_lambda, save_dir=save_dir)
            label = f"ewc_lambda={ewc_lambda:g}"
            save_classification_ewc_csv(steps, sgd_tr_cor_m, sgd_tr_cor_s, sgd_te_cor_m, sgd_te_cor_s, ewc_lambda,
                                        sgd_lr_fixed, hy_params_vanilla, save_dir)

            # train curve
            ax_tr.plot(steps, sgd_tr_cor_m, label=label)
            ax_tr.set_ylim(0, 1)
            ax_tr.fill_between(steps, sgd_tr_cor_m - sgd_tr_cor_s, sgd_tr_cor_m + sgd_tr_cor_s, alpha=0.2)

            # test curve
            ax_te.plot(steps, sgd_te_cor_m, label=label)
            ax_te.set_ylim(0, 1)
            ax_te.fill_between(steps, sgd_te_cor_m - sgd_te_cor_s, sgd_te_cor_m + sgd_te_cor_s, alpha=0.2)

            # Train MSE
            ax_tr_mse.plot(steps, sgd_tr_mse_m, label=label)
            ax_tr_mse.set_ylim(0, 1)
            ax_tr_mse.fill_between(steps, sgd_tr_mse_m - sgd_tr_mse_s, sgd_tr_mse_m + sgd_tr_mse_s, alpha=0.2)
            # test MSE
            ax_te_mse.plot(steps, sgd_te_mse_m, label=label)
            ax_te_mse.set_ylim(0, 1)
            ax_te_mse.fill_between(steps, sgd_te_mse_m - sgd_te_mse_s, sgd_te_mse_m + sgd_te_mse_s, alpha=0.2)
    else:
        hy_params_vanilla.update({'ewc_lambda': 0})
        for sgd_lr in sgd_lr_list:
            for gate_frac in gate_frac_list:
                print(f"parameters: lr={sgd_lr}, gate_frac={gate_frac}")
                steps, sgd_tr_cor_m, sgd_tr_cor_s, sgd_te_cor_m, sgd_te_cor_s, sgd_tr_mse_m, sgd_tr_mse_s, sgd_te_mse_m, sgd_te_mse_s = run_sgd_one_param(sgd_eta=1,
                                                                                                    sgd_gamma=1,
                                                                                                    sgd_lr=sgd_lr,
                                                                                                    hy_params_vanilla=hy_params_vanilla,
                                                                                                    label_correction_type=label_correction_type,
                                                                                                    order_type=hy_params_vanilla['train_order_type'], gate_frac=gate_frac, gate_mode=gate_mode, save_dir=save_dir)
                if gate_frac!=1:
                    save_classification_gating_csv(steps, sgd_tr_cor_m, sgd_tr_cor_s, sgd_te_cor_m, sgd_te_cor_s, gate_frac,
                                                   sgd_lr, hy_params_vanilla, gate_mode, save_dir)
                else:
                    save_classification_vanilla_csv(steps, sgd_tr_cor_m, sgd_tr_cor_s, sgd_te_cor_m, sgd_te_cor_s, sgd_lr,
                                                    hy_params_vanilla, epochs=epochs, mini_bz=z_mini_bz, save_dir=save_dir)

                if gate_mode=='none':
                    label = f"sgd_lr={sgd_lr:g}"
                else:
                    label = f"sgd_lr={sgd_lr:g}, gate_frac={gate_frac}"

                # train curve
                ax_tr.plot(steps, sgd_tr_cor_m, label=label)
                ax_tr.set_ylim(0, 1)
                ax_tr.fill_between(steps, sgd_tr_cor_m - sgd_tr_cor_s, sgd_tr_cor_m + sgd_tr_cor_s, alpha=0.2)

                # test curve
                ax_te.plot(steps, sgd_te_cor_m, label=label)
                ax_te.set_ylim(0, 1)
                ax_te.fill_between(steps, sgd_te_cor_m - sgd_te_cor_s, sgd_te_cor_m + sgd_te_cor_s, alpha=0.2)

                # Train MSE
                ax_tr_mse.plot(steps, sgd_tr_mse_m, label=label)
                ax_tr_mse.set_ylim(0, 1)
                ax_tr_mse.fill_between(steps, sgd_tr_mse_m - sgd_tr_mse_s, sgd_tr_mse_m + sgd_tr_mse_s, alpha=0.2)
                # test MSE
                ax_te_mse.plot(steps, sgd_te_mse_m, label=label)
                ax_te_mse.set_ylim(0, 1)
                ax_te_mse.fill_between(steps, sgd_te_mse_m - sgd_te_mse_s, sgd_te_mse_m + sgd_te_mse_s, alpha=0.2)

    ax_tr.legend(fontsize=8)
    ax_te.legend(fontsize=8)
    ax_tr_mse.legend(fontsize=8)
    ax_te_mse.legend(fontsize=8)
    fig_tr.tight_layout()
    fig_te.tight_layout()
    fig_tr_mse.tight_layout()
    fig_te_mse.tight_layout()

    fig_tr.savefig(fig_path1, dpi=200, bbox_inches="tight")
    fig_te.savefig(fig_path2, dpi=200, bbox_inches="tight")
    fig_tr_mse.savefig(fig_path_mse_tr, dpi=200, bbox_inches="tight")
    fig_te_mse.savefig(fig_path_mse_te, dpi=200, bbox_inches="tight")

def run_sgd_epochs_one_param(
        sgd_eta=0.1, sgd_gamma=0.1, sgd_lr=0.001, hy_params_vanilla=None, label_correction_type='online_correction',  # 'fixed_kernel_f0', 'online_correction', 'none'
        online_cor_type='iter',  # 'iter'/'full'
        order_type="random", gate_frac=1,  # gate fraction
        gate_mode="none",  # "infer", "none"
        ewc_lambda=0, epochs=10,
        per_seed_online_correction_csv_base_dir=None,
):
    hy_params = hy_params_vanilla.copy()
    hy_params.update({
        'gm': sgd_gamma,
        'eta': sgd_eta,
        'sgd_lr': sgd_lr,
        'gate_frac': gate_frac,
        'gate_mode': gate_mode,
        'ewc_lambda': ewc_lambda,
        'epochs': epochs
    })
    eval_every, device = hy_params['record_step'], hy_params['device']
    n_train, n_test, num_seeds = hy_params['Ntrain'], hy_params['Ntest'], hy_params['ikmax']
    # Curves over seeds
    train_accs_epochs_all, test_accs_epochs_all = [], []

    for sd in range(num_seeds):
        torch.manual_seed(sd)
        np.random.seed(sd)
        # Load dataset
        X_train_all, y_train_all, X_test, y_test, tr_idx, te_idx = load_ds(n_train, n_test, root="./data",
                                                                           seed_select=sd, device=device,
                                                                           hy_params=hy_params)

        # Order training stream
        order = make_order_multiclass(y_train_all, seed=sd, order_type=order_type, n_classes=10)
        X_train = X_train_all[order]
        y_train = y_train_all[order]
        Y_train_oh = one_hot(y_train, 10)
        X_test = X_test
        y_test = y_test
        # ----- (A) SGD baseline with iteratively corrected labels -----
        Niters = range(eval_every, n_train + 1, eval_every)
        if gate_mode=='none':
            use_gate=False
        else:
            use_gate=True

        if label_correction_type=='online_correction':
            model_sgd_both = SmallCNN_model(hy_params, use_gate).to(device)
            train_accs_epochs, test_accs_epochs = sgd_train_streaming_MSE_online_correction_epochs(model_sgd_both, Y_train_oh, X_train, y_train, X_test, y_test, Niters, hy_params, sgd_on_gm=sgd_gamma, sgd_on_eta=sgd_eta, online_cor_type=online_cor_type, fixed_first_ep_kernel=hy_params['fixed_first_ep_kernel'])
        elif label_correction_type=='none':
            model_sgd_van = SmallCNN_model(hy_params, use_gate).to(device)
            train_accs_epochs, test_accs_epochs = sgd_train_streaming_gating_ewc_epochs(model_sgd_van, X_train, y_train, X_test, y_test, Niters, hy_params, gate_mode, ewc_lambda)
        elif label_correction_type == 'cumulative':
            raise NotImplementedError(
                "cumulative is only supported for task_incremental with epochs==1 via "
                "run_sgd_params_sweep_plt / run_sgd_one_param (not run_sgd_epochs_one_param)."
            )
        else:
            train_accs_epochs, test_accs_epochs = [], []
            print("wrong label correction type")
        if (
            per_seed_online_correction_csv_base_dir is not None
            and label_correction_type == "online_correction"
            and len(train_accs_epochs) > 0
        ):
            epochs_x_seed = np.arange(1, hy_params["epochs"] + 1)
            save_sgd_online_correction_seed_csv(
                sgd_eta,
                sgd_gamma,
                sd,
                epochs_x_seed,
                train_accs_epochs,
                test_accs_epochs,
                per_seed_online_correction_csv_base_dir,
                on_cor_bs=hy_params.get("on_cor_bs", 0),
                fixed_first_K=hy_params.get("fixed_first_ep_kernel", True),
            )
        train_accs_epochs_all.append(train_accs_epochs)
        test_accs_epochs_all.append(test_accs_epochs)

    # Aggregate mean +/- SEM
    train_accs_epochs_m, train_accs_epochs_s = mean_and_sem(train_accs_epochs_all)
    test_accs_epochs_m, test_accs_epochs_s = mean_and_sem(test_accs_epochs_all)

    epochs_x = np.arange(1, hy_params['epochs'] + 1)
    return epochs_x, train_accs_epochs_m, train_accs_epochs_s, test_accs_epochs_m, test_accs_epochs_s

from plot import save_sgd_correction_csv
def run_sgd_epochs_params_sweep_plt(sgd_eta, sgd_gamma, sgd_lr_list, hy_params_vanilla, label_correction_type='none', gate_frac_list=[1], gate_mode='inferA', ewc_lambda_list=None, epochs=10, mini_batch_size=1, online_cor_type='batch_gpu'):
    # label_correction_type: 'fixed_kernel_f0', 'online_correction', 'none'
    if label_correction_type == 'cumulative':
        raise ValueError(
            "label_correction_type 'cumulative' is only for task_incremental + epochs==1; "
            "use run_sgd_params_sweep_plt, not run_sgd_epochs_params_sweep_plt."
        )
    num_seeds, Ntrain, sgd_mom, use2dense = hy_params_vanilla['ikmax'], hy_params_vanilla['Ntrain'], hy_params_vanilla['sgd_momentum'], hy_params_vanilla['use2dense']
    kernel_tp, z_mini_bz = hy_params_vanilla['kernel_tp'], hy_params_vanilla['z_mini_bz']
    on_cor_bs, z_type, first_ep_kernel = hy_params_vanilla['on_cor_bs'], hy_params_vanilla['z_type'], hy_params_vanilla['fixed_first_ep_kernel']
    train_order_type = hy_params_vanilla['train_order_type']
    hy_params_vanilla.update({'epochs': epochs})
    if label_correction_type not in ('none', 'cumulative'):
        save_dir = f"results_{hy_params_vanilla['ds_type']}_seeds{num_seeds}_N_tr{Ntrain}_2dense{use2dense}_epochs{epochs}/{train_order_type}_epochs{epochs}_sgd_bz{mini_batch_size}/sgd_correction/{label_correction_type}_order_{train_order_type}_{hy_params_vanilla['opt_type']}_mse_m{sgd_mom}_on_bs{on_cor_bs}_z{z_type}_z_bz{z_mini_bz}_fixed_first{first_ep_kernel}"
        fig_path1 = os.path.join(save_dir, f"train_curve_{kernel_tp}_record{hy_params_vanilla['record_step']}_mom{sgd_mom}.png")
        fig_path2 = os.path.join(save_dir, f"test_curve_sgd_{kernel_tp}_record{hy_params_vanilla['record_step']}_mom{sgd_mom}.png")
    elif ewc_lambda_list != None:
        save_dir = f"results_{hy_params_vanilla['ds_type']}_seeds{num_seeds}_N_tr{Ntrain}_2dense{use2dense}_epochs{epochs}/{train_order_type}_epochs{epochs}_sgd_bz{mini_batch_size}/sgd_ewc_every{hy_params_vanilla['ewc_update_every']}_{hy_params_vanilla['criterion']}/cor_{label_correction_type}_order_{train_order_type}_{hy_params_vanilla['opt_type']}_mom{sgd_mom}_on_bs{on_cor_bs}"
        fig_path1 = os.path.join(save_dir, f"train_acc_{kernel_tp}_record{hy_params_vanilla['record_step']}_mom{sgd_mom}.png")
        fig_path2 = os.path.join(save_dir, f"test_acc_{kernel_tp}_record{hy_params_vanilla['record_step']}_mom{sgd_mom}.png")
    else:
        save_dir = f"results_{hy_params_vanilla['ds_type']}_seeds{num_seeds}_N_tr{Ntrain}_2dense{use2dense}_epochs{epochs}/{train_order_type}_epochs{epochs}_sgd_bz{mini_batch_size}/sgd_{gate_mode}_{hy_params_vanilla['criterion']}_epochs{epochs}/label_vanilla_order_{train_order_type}_{hy_params_vanilla['opt_type']}_gate_{gate_mode}_mom{sgd_mom}_on_bs{on_cor_bs}"
        fig_path1 = os.path.join(save_dir, f"train_curve_batch{hy_params_vanilla['sgd_batch_size']}_{kernel_tp}_record{hy_params_vanilla['record_step']}_mom{sgd_mom}.png")
        fig_path2 = os.path.join(save_dir, f"test_curve_batch{hy_params_vanilla['sgd_batch_size']}_{kernel_tp}_record{hy_params_vanilla['record_step']}_mom{sgd_mom}.png")
    os.makedirs(save_dir, exist_ok=True)
    print("save_dir exists?", os.path.exists(save_dir), save_dir)
    print("abs save_dir len:", len(os.path.abspath(save_dir)))

    print("fig_path1:", fig_path1)
    print("abs fig_path1 len:", len(os.path.abspath(fig_path1)))
    # --- Train figure ---
    fig_tr, ax_tr = plt.subplots()
    ax_tr.set_xlabel("prefix length")
    ax_tr.set_ylabel("accuracy")
    ax_tr.grid(True, alpha=0.3)
    # --- Test figure ---
    fig_te, ax_te = plt.subplots()
    ax_te.set_xlabel("prefix length")
    ax_te.set_ylabel("accuracy")
    ax_te.grid(True, alpha=0.3)

    if label_correction_type not in ('none', 'cumulative'):
        ax_tr.set_title(f"Train acc, SGD + correction={label_correction_type}, Criterion={hy_params_vanilla['criterion']}, batch_size={hy_params_vanilla['sgd_batch_size']}")
        ax_te.set_title(f"Test acc, SGD + correction={label_correction_type}, Criterion={hy_params_vanilla['criterion']}, batch_size={hy_params_vanilla['sgd_batch_size']}")
    elif ewc_lambda_list != None:
        ax_tr.set_title(f"Train acc, SGD+EWC, SGD_fixed_lr, Criterion={hy_params_vanilla['criterion']}")
        ax_te.set_title(f"Test acc, SGD+EWC, SGD_fixed_lr, Criterion={hy_params_vanilla['criterion']}")
    else:
        ax_tr.set_title(f"Train acc, SGD, Gate={gate_mode}, Criterion={hy_params_vanilla['criterion']}, batch_size={hy_params_vanilla['sgd_batch_size']}")
        ax_te.set_title(f"Test acc, SGD, Gate={gate_mode}, Criterion={hy_params_vanilla['criterion']}, batch_size={hy_params_vanilla['sgd_batch_size']}")

    if label_correction_type not in ('none', 'cumulative'):
        print("using label correction")
        print(f"parameters: eta={sgd_eta}, gm={sgd_gamma}")
        epochs_x, train_accs_epochs_m, train_accs_epochs_s, test_accs_epochs_m, test_accs_epochs_s = run_sgd_epochs_one_param(sgd_eta=sgd_eta, sgd_gamma=sgd_gamma, sgd_lr=sgd_eta, hy_params_vanilla=hy_params_vanilla, label_correction_type=label_correction_type, online_cor_type=online_cor_type, order_type=hy_params_vanilla['train_order_type'], ewc_lambda=0, epochs=epochs, per_seed_online_correction_csv_base_dir=save_dir)
        label = f"lr={sgd_eta:g}, eta={sgd_eta:g}, gamma={sgd_gamma:g}"
        save_sgd_correction_csv(epochs_x, train_accs_epochs_m, train_accs_epochs_s, test_accs_epochs_m, test_accs_epochs_s, sgd_eta, sgd_gamma, save_dir)
        # train curve
        ax_tr.plot(epochs_x, train_accs_epochs_m, label=label)
        ax_tr.set_ylim(0, 1)
        ax_tr.fill_between(epochs_x, train_accs_epochs_m - train_accs_epochs_s,
                           train_accs_epochs_m + train_accs_epochs_s, alpha=0.2)

        # test curve
        ax_te.plot(epochs_x, test_accs_epochs_m, label=label)
        ax_te.set_ylim(0, 1)
        ax_te.fill_between(epochs_x, test_accs_epochs_m - test_accs_epochs_s,
                           test_accs_epochs_m + test_accs_epochs_s, alpha=0.2)
    elif gate_mode!='none':
        for sgd_lr in sgd_lr_list:
            for gate_frac in gate_frac_list:
                print(f"parameters: lr={sgd_lr}, gate_frac={gate_frac}")
                epochs_x, train_accs_epochs_m, train_accs_epochs_s, test_accs_epochs_m, test_accs_epochs_s = run_sgd_epochs_one_param(sgd_eta=1,
                                                                                                    sgd_gamma=1,
                                                                                                    sgd_lr=sgd_lr,
                                                                                                    hy_params_vanilla=hy_params_vanilla,
                                                                                                    label_correction_type=label_correction_type,
                                                                                                    order_type=hy_params_vanilla['train_order_type'], gate_frac=gate_frac, gate_mode=gate_mode, epochs=epochs)
                label = f"sgd_lr={sgd_lr:g}, gate_frac={gate_frac}"
                save_classification_gating_csv(epochs_x, train_accs_epochs_m, train_accs_epochs_s, test_accs_epochs_m, test_accs_epochs_s, gate_frac,
                                               sgd_lr, hy_params_vanilla, gate_mode, save_dir)
                # train curve
                ax_tr.plot(epochs_x, train_accs_epochs_m, label=label)
                ax_tr.set_ylim(0, 1)
                ax_tr.fill_between(epochs_x, train_accs_epochs_m - train_accs_epochs_s, train_accs_epochs_m + train_accs_epochs_s, alpha=0.2)

                # test curve
                ax_te.plot(epochs_x, test_accs_epochs_m, label=label)
                ax_te.set_ylim(0, 1)
                ax_te.fill_between(epochs_x, test_accs_epochs_m - test_accs_epochs_s, test_accs_epochs_m + test_accs_epochs_s, alpha=0.2)
    else:
        for sgd_lr in sgd_lr_list:
            print(f"parameters: lr={sgd_lr}")
            epochs_x, train_accs_epochs_m, train_accs_epochs_s, test_accs_epochs_m, test_accs_epochs_s = run_sgd_epochs_one_param(sgd_eta=1,
                                                                                                sgd_gamma=1,
                                                                                                sgd_lr=sgd_lr,
                                                                                                hy_params_vanilla=hy_params_vanilla,
                                                                                                label_correction_type=label_correction_type,
                                                                                                order_type=hy_params_vanilla['train_order_type'], gate_frac=1, gate_mode=gate_mode, ewc_lambda=0, epochs=epochs)
            label = f"sgd_lr={sgd_lr:g}"
            save_classification_vanilla_csv(epochs_x, train_accs_epochs_m, train_accs_epochs_s, test_accs_epochs_m, test_accs_epochs_s,
                                            sgd_lr, hy_params_vanilla, epochs=epochs, mini_bz=mini_batch_size, save_dir=save_dir)

            # train curve
            ax_tr.plot(epochs_x, train_accs_epochs_m, label=label)
            ax_tr.set_ylim(0, 1)
            ax_tr.fill_between(epochs_x, train_accs_epochs_m - train_accs_epochs_s, train_accs_epochs_m + train_accs_epochs_s, alpha=0.2)

            # test curve
            ax_te.plot(epochs_x, test_accs_epochs_m, label=label)
            ax_te.set_ylim(0, 1)
            ax_te.fill_between(epochs_x, test_accs_epochs_m - test_accs_epochs_s, test_accs_epochs_m + test_accs_epochs_s, alpha=0.2)

    ax_tr.legend(fontsize=8)
    ax_te.legend(fontsize=8)
    fig_tr.tight_layout()
    fig_te.tight_layout()

    fig_tr.savefig(fig_path1, dpi=200, bbox_inches="tight")
    fig_te.savefig(fig_path2, dpi=200, bbox_inches="tight")

def run_offline_one_param(
    order_type="random",  # "random" or "class_incremental"
    gamma=1,
    hy_params_vanilla=None
):
    hy_params = hy_params_vanilla.copy()
    hy_params.update({
        'gm': gamma,
    })
    device, eval_every = hy_params['device'], hy_params['record_step']
    kernel_tp, bias_exist = hy_params['kernel_tp'], hy_params['bias_exist']
    n_train, n_test, gamma, num_seeds = hy_params['Ntrain'], hy_params['Ntest'], hy_params['gm'], hy_params['ikmax']

    # Steps to evaluate
    steps = list(range(eval_every, n_train + 1, eval_every))

    # Curves over seeds
    krr_tr_all, krr_te_all = [], []

    for sd in range(num_seeds):
        print(f"\n=== Seed {sd} | order={order_type} ===")
        torch.manual_seed(sd)
        np.random.seed(sd)
        # Load dataset
        X_train_all, y_train_all, X_test_all, y_test_all, tr_idx, te_idx = load_ds(n_train, n_test, root="./data", seed_select=sd, device=device, hy_params=hy_params)

        # Order training stream
        train_order = make_order_multiclass(y_train_all, seed=sd, order_type=order_type, n_classes=10)
        X_train = X_train_all[train_order]
        y_train = y_train_all[train_order]
        # Order test stream
        test_order = make_order_multiclass(y_test_all, seed=sd, order_type=order_type, n_classes=10)
        X_test = X_test_all[test_order]
        y_test = y_test_all[test_order]
        torch.manual_seed(sd)
        model_ntk = SmallCNN_model(hy_params, use_gate=False).to(device)
        K_train, K_tr_te = kernel_type(model_ntk, X_train, X_test, device, kernel_tp, hy_params)

        Y_train_oh = one_hot(y_train, 10)
        #print("Yoh.shape", Y_train_oh.shape)

        # Accuracy vs steps for kernel methods
        krr_tr, krr_te = [], []
        # offline learning here, try to pick the kernel prefix as the index
        for s in steps:
            Yd = Y_train_oh[:s]
            K_dd, K_dq_tr, K_dq_te, y_test_sub, y_test_sub_oh = step_kernel(s, y_train, y_test, K_train, K_tr_te, hy_params)

            # KRR baseline
            if hy_params['Class_wise']:
                print("class wise kernel is used")
                _, yhat_tr = krr_predict_classwise(K_dd, K_dq_tr, Yd, gamma=gamma)
                _, yhat_te = krr_predict_classwise(K_dd, K_dq_te, Yd, gamma=gamma)
            else:
                _, yhat_tr = krr_predict_precomputed(K_dd, K_dq_tr, Yd, gamma=gamma)
                _, yhat_te = krr_predict_precomputed(K_dd, K_dq_te, Yd, gamma=gamma)
            krr_tr.append(np.mean(yhat_tr == y_train[:s]))
            krr_te.append(np.mean(yhat_te == y_test_sub))

        krr_tr_all.append(krr_tr); krr_te_all.append(krr_te)

    # Aggregate mean +/- SEM
    krr_tr_m, krr_tr_s = mean_and_sem(krr_tr_all)
    krr_te_m, krr_te_s = mean_and_sem(krr_te_all)
    return steps, krr_tr_m, krr_tr_s, krr_te_m, krr_te_s


def step_kernel_task_incremental(s, y_train, y_test, K_train, K_tr_te, hy_params, num_tasks=5, num_classes_per_task=2):
    num_train_per_task = len(y_train) // num_tasks
    num_test_per_task = len(y_test) // num_tasks

    tasks_seen = int((s - 1) // num_train_per_task) + 1

    t = tasks_seen * num_test_per_task

    if hy_params.get('Class_wise', False):
        K_dd = K_train[:, :s, :s]
        K_dq_tr = K_train[:, :s, :s]
        K_dq_te = K_tr_te[:, :s, :t]
    else:
        K_dd = K_train[:s, :s]
        K_dq_tr = K_train[:s, :s]
        K_dq_te = K_tr_te[:s, :t]

    y_test_sub = y_test[:t]
    y_test_sub_oh = np.eye(num_classes_per_task)[y_test_sub]

    return K_dd, K_dq_tr, K_dq_te, y_test_sub, y_test_sub_oh

import pandas as pd
def run_offline_multiple_gammas(gammas=[0.1, 1.0, 10.0],
                                hy_params_vanilla=None):
    hy_params = hy_params_vanilla.copy()

    device = hy_params['device']
    eval_every = hy_params['record_step']
    kernel_tp = hy_params['kernel_tp']

    order_type = hy_params['train_order_type']
    n_train = hy_params['Ntrain']
    n_test = hy_params['Ntest']
    num_seeds = hy_params['ikmax']

    train_order_type = hy_params_vanilla['train_order_type']
    if hy_params_vanilla['Class_wise']:
        save_dir = f"results_{hy_params_vanilla['ds_type']}_N_tr{hy_params_vanilla['Ntrain']}_{order_type}/offline_classwise_kernel_order_{train_order_type}"
    else:
        save_dir = f"results_{hy_params_vanilla['ds_type']}_N_tr{hy_params_vanilla['Ntrain']}_{order_type}/offline_no_classwise_no_rescale_order_{train_order_type}"
    os.makedirs(save_dir, exist_ok=True)

    # Steps to evaluate
    steps = list(range(eval_every, n_train + 1, eval_every))
    if steps[-1] != n_train:
        steps.append(n_train)

    # Dictionary to hold the raw results for all seeds, separated by gamma
    gamma_results = {gm: {'tr_all': [], 'te_all': []} for gm in gammas}

    for sd in range(num_seeds):
        print(f"\n=== Seed {sd} | order={order_type} ===")
        torch.manual_seed(sd)
        np.random.seed(sd)
        # Load dataset
        X_train_all, y_train_all, X_test_all, y_test_all, tr_idx, te_idx = load_ds(n_train, n_test, root="./data",
                                                                                   seed_select=sd, device=device,
                                                                                   hy_params=hy_params)
        # Order training stream
        if hy_params_vanilla['train_order_type'] == 'task_incremental':
            X_train, y_train, X_test, y_test, task_pairs = format_split_binary_tasks(X_train_all, y_train_all, X_test_all, y_test_all, seed=sd)
            Y_train_oh = one_hot(y_train, 2)
        else:
            order = make_order_multiclass(y_train_all, seed=sd, order_type=order_type, n_classes=10)
            X_train = X_train_all[order]
            y_train = y_train_all[order]
            Y_train_oh = one_hot(y_train,10)
            X_test, y_test = X_test_all, y_test_all

        model_ntk = SmallCNN_model(hy_params, use_gate=False).to(device)

        # Compute Kernel Matrices ONCE per seed
        K_train, K_tr_te = kernel_type(model_ntk, X_train, X_test, device, kernel_tp, hy_params)

        # Temp tracking for the current seed's steps, split by gamma
        seed_tr = {gm: [] for gm in gammas}
        seed_te = {gm: [] for gm in gammas}

        # Offline learning step loop
        for s in steps:
            Yd = Y_train_oh[:s]
            num_train_per_task = len(y_train)/5
            task_id = int(s/num_train_per_task)
            # Slice matrices ONCE per step
            if order_type=='random' or order_type=='class_incremental':
                K_dd, K_dq_tr, K_dq_te, y_test_sub, y_test_sub_oh = step_kernel(s, y_train, y_test, K_train, K_tr_te,
                                                                                hy_params)
            elif order_type=='task_incremental':
                K_dd, K_dq_tr, K_dq_te, y_test_sub, y_test_sub_oh = step_kernel_task_incremental(s, y_train, y_test,
                                                                                                 K_train, K_tr_te, hy_params,
                                                                                                 num_tasks=5, num_classes_per_task=2)
            else:
                print("incorrect order type")
            # Innermost loop: Solve for each gamma
            for gm in gammas:
                # Dynamically update gamma in hy_params if step_kernel or others need it
                hy_params['gm'] = gm

                if hy_params['Class_wise']:
                    if s == steps[0] and sd == 0 and gm == gammas[0]:  # Print just once to avoid spam
                        print("class wise kernel is used")
                    _, yhat_tr = krr_predict_classwise(K_dd, K_dq_tr, Yd, gamma=gm)
                    _, yhat_te = krr_predict_classwise(K_dd, K_dq_te, Yd, gamma=gm)
                else:
                    _, yhat_tr = krr_predict_precomputed(K_dd, K_dq_tr, Yd, gamma=gm)
                    _, yhat_te = krr_predict_precomputed(K_dd, K_dq_te, Yd, gamma=gm)

                seed_tr[gm].append(np.mean(yhat_tr == y_train[:s]))
                seed_te[gm].append(np.mean(yhat_te == y_test_sub))

        for gm in gammas:
            gamma_results[gm]['tr_all'].append(seed_tr[gm])
            gamma_results[gm]['te_all'].append(seed_te[gm])

    # Aggregate mean +- SEM and save to CSV
    final_output = {}
    for gm in gammas:
        # Cast the list of lists into a 2D NumPy array
        tr_arr = np.array(gamma_results[gm]['tr_all'])
        te_arr = np.array(gamma_results[gm]['te_all'])

        tr_m, tr_s = mean_and_sem(tr_arr)
        te_m, te_s = mean_and_sem(te_arr)

        # Save to a dictionary if you still need to return them in memory
        final_output[gm] = {'tr_m': tr_m, 'tr_s': tr_s, 'te_m': te_m, 'te_s': te_s}

        df = pd.DataFrame({
            'step': steps,
            'train_mean': tr_m,
            'train_sem': tr_s,
            'test_mean': te_m,
            'test_sem': te_s,
            'gamma': gm
        })

        csv_filename = os.path.join(save_dir, f"krr_results_N_tr{hy_params['Ntrain']}_gamma_{gm}.csv")
        df.to_csv(csv_filename, index=False)


from plot import save_off_reg_csv
def run_offline_params_sweep_plt(gamma_list, hy_params_vanilla):
    train_order_type = hy_params_vanilla['train_order_type']
    if hy_params_vanilla['Class_wise']:
        save_dir = f"results_{hy_params_vanilla['ds_type']}_N_tr{hy_params_vanilla['Ntrain']}/offline_classwise_kernel_order_{train_order_type}"
    else:
        save_dir = f"results_{hy_params_vanilla['ds_type']}_N_tr{hy_params_vanilla['Ntrain']}/offline_no_classwise_no_rescale_order_{train_order_type}"
    fig_path1 = os.path.join(save_dir, f"train_curve.png")
    fig_path2 = os.path.join(save_dir, f"test_curve.png")
    os.makedirs(save_dir, exist_ok=True)

    # --- Train figure ---
    fig_tr, ax_tr = plt.subplots()
    ax_tr.set_xlabel("prefix length")
    ax_tr.set_ylabel("accuracy")
    ax_tr.grid(True, alpha=0.3)
    # --- Test figure ---
    fig_te, ax_te = plt.subplots()
    ax_te.set_xlabel("prefix length")
    ax_te.set_ylabel("accuracy")
    ax_te.grid(True, alpha=0.3)

    ax_tr.set_title(f"Train accuracy (offline)")
    ax_te.set_title(f"Test accuracy (offline)")

    for gm in gamma_list:
        print(f"gm={gm}")
        steps, krr_tr_m, krr_tr_s, krr_te_m, krr_te_s = run_offline_one_param(order_type=train_order_type, gamma=gm, hy_params_vanilla=hy_params_vanilla)
        label = f"gamma={gm}"
        save_off_reg_csv(steps, krr_tr_m, krr_tr_s, krr_te_m, krr_te_s, gm, hy_params_vanilla, save_dir=save_dir)
        # train curve
        ax_tr.plot(steps, krr_tr_m, label=label)
        ax_tr.set_ylim(0, 1)
        ax_tr.fill_between(steps, krr_tr_m - krr_tr_s, krr_tr_m + krr_tr_s, alpha=0.2)

        # test curve
        ax_te.plot(steps, krr_te_m, label=label)
        ax_te.set_ylim(0, 1)
        ax_te.fill_between(steps, krr_te_m - krr_te_s, krr_te_m + krr_te_s, alpha=0.2)

    ax_tr.legend(fontsize=8)
    ax_te.legend(fontsize=8)
    fig_tr.tight_layout()
    fig_te.tight_layout()

    # after you plot each figure:
    fig_tr.savefig(fig_path1, dpi=200, bbox_inches="tight")
    # ... make another figure ...
    fig_te.savefig(fig_path2, dpi=200, bbox_inches="tight")



def main_epochs(args, hy_params_vanilla, label_correction_tp, epochs, sgd_van_lr, ewc_lambda, online_cor_type='batch_gpu'):
    N_train, n_seeds = hy_params_vanilla['Ntrain'], hy_params_vanilla['ikmax']
    kernel_tp = hy_params_vanilla['kernel_tp']
    on_cor_bs = hy_params_vanilla['on_cor_bs']
    train_order_type = hy_params_vanilla['train_order_type']
    if label_correction_tp == 'online_correction':
        if online_cor_type != 'batch_gpu':
            on_cor_bs = 0
        online_csv_dir = f'results/cifar10_{train_order_type}_Ntr{N_train}_seed{n_seeds}_{kernel_tp}/sgd_online_{online_cor_type}_b{on_cor_bs}_correction_csv'
        (epochs_x, tr_m, tr_s, te_m, te_s) = run_sgd_epochs_one_param(sgd_eta=args.sgd_eta, sgd_gamma=args.sgd_gamma,
                                                                      sgd_lr=args.sgd_eta,  # args.sgd_eta,
                                                                      hy_params_vanilla=hy_params_vanilla,
                                                                      label_correction_type='online_correction',
                                                                      online_cor_type=online_cor_type, # 'full', 'iter', 'batch_gpu'
                                                                      order_type=hy_params_vanilla['train_order_type'],
                                                                      epochs=epochs, ewc_lambda=0,
                                                                      per_seed_online_correction_csv_base_dir=online_csv_dir)
        save_sgd_online_correction_csv(args.sgd_eta, args.sgd_gamma, epochs_x, tr_m, tr_s, te_m, te_s,
                                       output_dir=online_csv_dir, on_cor_bs=on_cor_bs)
    elif label_correction_tp == 'none':
        if ewc_lambda>0:
            (epochs_x, tr_m, tr_s, te_m, te_s) = run_sgd_epochs_one_param(sgd_eta=args.sgd_eta, sgd_gamma=args.sgd_gamma,
                                                                          sgd_lr=sgd_van_lr,  # args.sgd_eta,
                                                                          hy_params_vanilla=hy_params_vanilla,
                                                                          label_correction_type='none',
                                                                          order_type=hy_params_vanilla['train_order_type'],
                                                                          epochs=epochs, ewc_lambda=ewc_lambda)
            save_sgd_ewc_csv(sgd_van_lr, ewc_lambda, epochs_x, tr_m, tr_s, te_m, te_s, output_dir=f'results/cifar10_{train_order_type}_Ntr{N_train}_seed{n_seeds}_{kernel_tp}/sgd_ewc_csv')
        else:
            # vanilla sgd
            (epochs_x, tr_m, tr_s, te_m, te_s) = run_sgd_epochs_one_param(sgd_eta=args.sgd_eta,
                                                                          sgd_gamma=args.sgd_gamma, sgd_lr=sgd_van_lr, # args.sgd_eta,
                                                                          hy_params_vanilla=hy_params_vanilla,
                                                                          label_correction_type='none',
                                                                          order_type=hy_params_vanilla['train_order_type'], epochs=epochs, ewc_lambda=ewc_lambda)
            save_sgd_vanilla_csv(sgd_lr=sgd_van_lr, epochs_x=epochs_x, tr_m=tr_m, tr_s=tr_s, te_m=te_m, te_s=te_s,
                                 output_dir=f'results/cifar10_{train_order_type}_Ntr{N_train}_seed{n_seeds}_{kernel_tp}/sgd_vanilla_csv')
    else:
        print("incorrect parameters setting")


if __name__ == "__main__":
    def str2bool(v):
        return v.lower() in ('true', '1', 'yes')

    parser = argparse.ArgumentParser(description='Parallel Hyperparameter Sweep')
    # --- label correction hyperparameters ---
    parser.add_argument('--sgd_eta',   type=float, default=0.01,  help='Online learning rate eta')
    parser.add_argument('--sgd_gamma', type=float, default=100,   help='KRR regularisation gamma')
    # --- network ---
    parser.add_argument('--use2dense',  type=str2bool, default=False,   help='Add hidden dense layer (True/False)')
    parser.add_argument('--ds_type',    type=str,      default='cifar10', help='Dataset: cifar10 or mnist')
    # --- training order & label correction type ---
    parser.add_argument('--train_order_type',      type=str,      default='task_incremental',
                        help='Stream order: random / class_incremental / task_incremental')
    parser.add_argument('--z_type',                type=str,      default='both',
                        help='Label correction type: both / online / offline')
    parser.add_argument('--fixed_first_ep_kernel', type=str2bool, default=False,
                        help='Fix kernel after first epoch (True/False)')
    # --- training schedule ---
    parser.add_argument('--adap_kernel_times', type=int, default=1,
                        help='Number of kernel updates (= number of tasks in task-incremental)')
    parser.add_argument('--epochs',        type=int, default=1,     help='Training epochs per task')
    parser.add_argument('--ikmax',         type=int, default=5,     help='Number of random seeds')
    parser.add_argument('--Ntrain',        type=int, default=30000, help='Training set size')
    parser.add_argument('--Ntest',         type=int, default=10000, help='Test set size')
    # --- label correction batching ---
    parser.add_argument('--on_cor_bs',      type=int, default=20,   help='Label correction batch size')
    parser.add_argument('--sgd_batch_size', type=int, default=1,    help='SGD mini-batch size')
    parser.add_argument('--z_mini_bz',      type=int, default=1,    help='Mini-batch size for KbU masking')
    parser.add_argument('--label_correction_type', type=str, default='online_correction',
                        help='online_correction | none | cumulative (task_incr+epochs1: vanilla + cumulative replay; use --replay_schedule)')
    parser.add_argument('--sgd_lr_list', type=float, nargs='+',
                        default=[0.0001, 0.0003, 0.001, 0.003, 0.01, 0.02, 0.05],
                        help='SGD learning rates to sweep (space-separated)')
    parser.add_argument('--online_cor_type', type=str, default='batch_gpu',
                        help='Online correction compute mode: batch_gpu / iter / full')
    parser.add_argument('--record_step', type=int, default=500,
                        help='Evaluation interval; must divide Ntrain/adap_kernel_times')
    parser.add_argument('--replay_schedule', type=str, default='task_boundary',
                        choices=['task_boundary', 'every_chunk'],
                        help='cumulative only: when to run shuffled replay over X_train[:N] - '
                             'task_boundary (default) or every_chunk (after each record_step; much slower)')
    parser.add_argument(
        '--EWC_LIST',
        type=float,
        nargs='*',
        default=None,
        help='task_incremental + none: pass one or more ewc_lambda values to sweep SGD+EWC; omit flag for vanilla.',
    )
    parser.add_argument(
        '--SGD_LR_FIXED',
        type=float,
        default=0.01,
        help='Fixed SGD lr passed to run_sgd_params_sweep_plt (vanilla and EWC).',
    )

    args = parser.parse_args()
    hy_params_vanilla = {
        'device': torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        # network setting
        'use2dense': args.use2dense,
        'use_resnet': False,

        'ds_type': args.ds_type,
        'train_order_type': args.train_order_type, # 'random'/'class_incremental'/'task_incremental'
        'opt_type': 'sgd',  # 'adam', 'sgd'
        'criterion': 'mse',  # 'mse', 'crossentropy'
        'z_type': args.z_type,
        'fixed_first_ep_kernel': args.fixed_first_ep_kernel, # for random multiple epochs

        'adap_kernel_times': args.adap_kernel_times, # for task incremental
        'epochs': args.epochs,
        'ikmax': args.ikmax,
        'Ntrain': args.Ntrain,
        'Ntest': args.Ntest,

        'online_cor_type': args.online_cor_type, # 'online_correction','none','cumulative'
        'on_cor_bs': args.on_cor_bs,
        'KbU_block_size': 1,  # block-wise size for online KbU: 16, 32
        'record_step': args.record_step,
        'replay_schedule': args.replay_schedule,
        'Nb': 16,            # block size for memory efficiency

        'gm': args.sgd_gamma,
        'eta': args.sgd_eta,
        'sgd_lr': 0.005,
        'sgd_momentum': 0,
        'sgd_weight_decay': 0,
        'sgd_batch_size': args.sgd_batch_size,
        'z_mini_bz': args.z_mini_bz,

        'kernel_tp': "jacobian_empirical_no_rescale", # 'feature_kernel'/"jacobian_empirical_no_rescale"/"vjp_jvp_no_rescale", "stax_kernel", "jacobian_empirical_rescale"#  "jacobian_norm", 'jacobian_classwise_no_rescale'
        'Class_wise': False,  # class-wise kernel calculation
        'Class_avg': True,  # average over class number in kernel calculation
        # 'train_kernel_update': True, # update the mini-batch kernel calculation for train kernel matrix

        "bias_conv": False,  # True or False
        "bias_dense": False,  # True or False
        "stax_bias_conv": None,  # float or None
        "stax_bias_dense": None,  # float or None
        "bias_exist": "none", # "full"(bias_conv&bias_dense=True/float), "conv"(only bias_conv=True/float), "none"(bias_conv&bias_dense=False/None)

        # gate
        'gate_frac': 1,
        'gate_mode': 'none',  # test gate_mode = 'none', 'oracle', 'inferA', 'inferA_self', 'inferB_max', 'inferB_lse', "infer_entropy"
        "lambda_loss_gate": 0,
        # EWC
        "ewc_lambda": 0,  # 0 if not using ewc
        "ewc_update_every": 100,  # N_train/10
        "ewc_fisher_bs": 32,  # batch when calculating fisher
        "ewc_fisher_n": 10000,  # subsample for fisher matrix(optional)

    }
    Begin = time.time()

    print("on cor bs is:", hy_params_vanilla['on_cor_bs'], "eta is:", args.sgd_eta, "gamma is:", args.sgd_gamma)

    alpha_list = [3]
    gammas = [0.01, 0.1, 1, 10, 30, 100, 300]
    sgd_eta_list, sgd_gamma_list = [0.003, 0.01, 0.03], [30, 100, 300]
    gate_frac_list = [0.1, 0.3, 0.5, 0.7, 0.999]
    mini_bz_list = [1, 5, 10, 20, 50]

    sgd_lr_list = args.sgd_lr_list
    label_correction_type = args.label_correction_type

    if hy_params_vanilla['train_order_type'] == 'random':
        if hy_params_vanilla['epochs']==1:
            if label_correction_type == 'online_correction':
                hy_params_vanilla.update({'fixed_first_ep_kernel': True})
                run_sgd_params_sweep_plt(args.sgd_eta, args.sgd_gamma, sgd_lr_list, sgd_lr_fixed=args.SGD_LR_FIXED,
                                        hy_params_vanilla=hy_params_vanilla, label_correction_type='online_correction',
                                        gate_frac_list=[1], gate_mode='none', ewc_lambda_list=None, online_cor_type=hy_params_vanilla['online_cor_type'])
                hy_params_vanilla.update({'fixed_first_ep_kernel': False})
                run_sgd_params_sweep_plt(args.sgd_eta, args.sgd_gamma, sgd_lr_list, sgd_lr_fixed=args.SGD_LR_FIXED,
                                        hy_params_vanilla=hy_params_vanilla, label_correction_type='online_correction',
                                        gate_frac_list=[1], gate_mode='none', ewc_lambda_list=None, online_cor_type=hy_params_vanilla['online_cor_type'])
            else:
                run_sgd_params_sweep_plt(args.sgd_eta, args.sgd_gamma, sgd_lr_list, sgd_lr_fixed=args.SGD_LR_FIXED,
                                        hy_params_vanilla=hy_params_vanilla, label_correction_type=label_correction_type,
                                        gate_frac_list=[1], gate_mode='none', ewc_lambda_list=None, online_cor_type=hy_params_vanilla['online_cor_type'])
        else:
            if label_correction_type == 'online_correction':
                hy_params_vanilla.update({'fixed_first_ep_kernel': False})
                run_sgd_epochs_params_sweep_plt(args.sgd_eta, args.sgd_gamma, sgd_lr_list, hy_params_vanilla,
                                                label_correction_type=label_correction_type,
                                                gate_frac_list=[1], gate_mode='none', ewc_lambda_list=None,
                                                epochs=hy_params_vanilla['epochs'],
                                                mini_batch_size=hy_params_vanilla['sgd_batch_size'],
                                                online_cor_type=hy_params_vanilla['online_cor_type'])
            else:
                run_sgd_epochs_params_sweep_plt(args.sgd_eta, args.sgd_gamma, sgd_lr_list, hy_params_vanilla, label_correction_type=label_correction_type,
                                gate_frac_list=[1], gate_mode='none', ewc_lambda_list=None, epochs=hy_params_vanilla['epochs'],
                                mini_batch_size=hy_params_vanilla['sgd_batch_size'], online_cor_type=hy_params_vanilla['online_cor_type'])
    elif hy_params_vanilla['train_order_type'] == 'task_incremental':
        if label_correction_type == 'online_correction':
            hy_params_vanilla.update({'adap_kernel_times': 5})
            run_sgd_params_sweep_plt(args.sgd_eta, args.sgd_gamma, sgd_lr_list, sgd_lr_fixed=args.SGD_LR_FIXED,
                                    hy_params_vanilla=hy_params_vanilla, label_correction_type=label_correction_type,
                                    gate_frac_list=[1], gate_mode='none', ewc_lambda_list=None, online_cor_type=hy_params_vanilla['online_cor_type'])
            hy_params_vanilla.update({'adap_kernel_times': 1})
            run_sgd_params_sweep_plt(args.sgd_eta, args.sgd_gamma, sgd_lr_list, sgd_lr_fixed=args.SGD_LR_FIXED,
                                    hy_params_vanilla=hy_params_vanilla, label_correction_type=label_correction_type,
                                    gate_frac_list=[1], gate_mode='none', ewc_lambda_list=None, online_cor_type=hy_params_vanilla['online_cor_type'])
        elif label_correction_type == 'cumulative':
            if hy_params_vanilla['epochs'] != 1:
                raise ValueError("cumulative with task_incremental requires epochs==1")
            run_sgd_params_sweep_plt(
                args.sgd_eta, args.sgd_gamma, sgd_lr_list, sgd_lr_fixed=args.SGD_LR_FIXED,
                hy_params_vanilla=hy_params_vanilla,
                label_correction_type='cumulative',
                gate_frac_list=[1], gate_mode='none', ewc_lambda_list=None,
                online_cor_type=hy_params_vanilla['online_cor_type'],
            )
        elif label_correction_type == 'none':
            if hy_params_vanilla['epochs'] != 1:
                raise ValueError("task_incremental vanilla (none): use epochs==1 with run_sgd_params_sweep_plt")
            ewc_lambda_list = None if not args.EWC_LIST else args.EWC_LIST
            run_sgd_params_sweep_plt(
                args.sgd_eta, args.sgd_gamma, sgd_lr_list, sgd_lr_fixed=args.SGD_LR_FIXED,
                hy_params_vanilla=hy_params_vanilla,
                label_correction_type='none',
                gate_frac_list=[1], gate_mode='none', ewc_lambda_list=ewc_lambda_list,
                online_cor_type=hy_params_vanilla['online_cor_type'],
            )
        else:
            raise ValueError(
                f"Unsupported label_correction_type for task_incremental: {label_correction_type!r} "
                "(expected 'online_correction', 'cumulative', or 'none')."
            )
    elif hy_params_vanilla['train_order_type'] == 'class_incremental':
        gamma_list = [0.1, 0.3, 1, 3, 10, 30]
        eta_list, gamma_list = [0.0003, 0.001, 0.003, 0.01, 0.03, 0.1], [1]
        gamma_list = [0.1, 1, 10]
        eta_list = [0.001, 0.01, 0.1]
        visualize_label_correction_class_incremental(
            hy_params_vanilla=hy_params_vanilla,
            eta=0.01,
            gamma=200,
            online_cor_type='iter',
            seed=0
        )

    End = time.time()
    print("time cost:", End-Begin)