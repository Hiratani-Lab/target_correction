import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import glob

def save_summary_csv(
    out_csv,
    steps,
    series_dict,
    meta=None,
):
    steps = np.asarray(steps)
    rows = []

    for name, (m, s) in series_dict.items():
        print(name)
        m = np.asarray(m)
        s = np.asarray(s)
        assert m.shape == steps.shape, f"{name} mean shape {m.shape} != steps {steps.shape}"
        assert s.shape == steps.shape, f"{name} sem shape {s.shape} != steps {steps.shape}"

        split = "train" if name.endswith("_tr") else ("test" if name.endswith("_te") else "unknown")
        method = name.replace("_tr", "").replace("_te", "")

        for step, mean, sem in zip(steps, m, s):
            row = {
                "step": int(step),
                "split": split,
                "method": method,
                "mean": float(mean),
                "sem": float(sem),
                "series": name,  # keeps your original naming
            }
            if meta:
                row.update(meta)
            rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print(f"Saved summary CSV -> {out_csv} ({len(df)} rows)")
    return df


def plot_summary_csv(
    csv_path,
    split="train",          # "train" or "test"
    use_sem=True,
    title=None,
    ylim=(0.0, 1.05),
    legend_fontsize=8,
    save_fig_path='lr',
):
    df = pd.read_csv(csv_path)
    df = df[df["split"] == split].copy()

    plt.figure()
    for method, g in df.groupby("method"):
        g = g.sort_values("step")
        x = g["step"].to_numpy()
        y = g["mean"].to_numpy()
        plt.plot(x, y, label=method)

        if use_sem:
            se = g["sem"].to_numpy()
            plt.fill_between(x, y - se, y + se, alpha=0.2)

    plt.xlabel("Number of training samples used (prefix length)")
    plt.ylabel("Accuracy")
    if ylim is not None:
        plt.ylim(*ylim)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=legend_fontsize)

    if title is None:
        title = f"Accuracy ({split})"
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_fig_path, dpi=200)
    plt.show()
    plt.close()


def save_sgd_online_correction_csv(sgd_eta, sgd_gm, epochs_x, tr_m, tr_s, te_m, te_s, output_dir='results_cifar10_random/sgd_online_correction_csv', on_cor_bs=0):
    # 1. Create a DataFrame where each row is one epoch
    df = pd.DataFrame({
        'epoch': epochs_x,
        'sgd_eta': sgd_eta,
        'sgd_gamma': sgd_gm,
        'train_acc_mean': tr_m,
        'train_acc_std': tr_s,
        'test_acc_mean': te_m,
        'test_acc_std': te_s
    })

    # 3. Save to CSV
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    file_name = f"res_eta_{sgd_eta}_gm_{sgd_gm}_bs_{on_cor_bs}.csv"
    save_path = os.path.join(output_dir, file_name)
    df.to_csv(save_path, index=False)
    print(f"Saved results for Eta={sgd_eta}, Gamma={sgd_gm} to {save_path}")


def save_sgd_online_correction_seed_csv(
    sgd_eta,
    sgd_gm,
    seed,
    epochs_x,
    train_acc,
    test_acc,
    output_base_dir,
    on_cor_bs=0,
    fixed_first_K=True,
):
    epochs_x = np.asarray(epochs_x, dtype=float)
    train_acc = np.asarray(train_acc, dtype=float)
    test_acc = np.asarray(test_acc, dtype=float)
    if train_acc.shape != epochs_x.shape or test_acc.shape != epochs_x.shape:
        raise ValueError(
            f"epochs_x shape {epochs_x.shape} must match train_acc {train_acc.shape} "
            f"and test_acc {test_acc.shape}"
        )

    sub = f"cor_seed{seed}_bs_{on_cor_bs}_fixed_first_K{fixed_first_K}"
    out_dir = os.path.join(output_base_dir, sub)
    os.makedirs(out_dir, exist_ok=True)

    df = pd.DataFrame(
        {
            "epoch": epochs_x,
            "sgd_eta": sgd_eta,
            "sgd_gamma": sgd_gm,
            "seed": seed,
            "train_acc": train_acc,
            "test_acc": test_acc,
        }
    )
    file_name = f"res_eta_{sgd_eta}_gm_{sgd_gm}_seed{seed}_bs_{on_cor_bs}.csv"
    save_path = os.path.join(out_dir, file_name)
    df.to_csv(save_path, index=False)
    print(f"Saved per-seed results (seed={seed}) to {save_path}")

def save_sgd_vanilla_csv(sgd_lr, epochs_x, tr_m, tr_s, te_m, te_s, output_dir='results_cifar10_random/sgd_vanilla_csv'):
    # 1. Create a DataFrame where each row is one epoch
    df = pd.DataFrame({
        'epoch': epochs_x,
        'sgd_lr': sgd_lr,
        'train_acc_mean': tr_m,
        'train_acc_std': tr_s,
        'test_acc_mean': te_m,
        'test_acc_std': te_s
    })

    # 3. Save to CSV
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    file_name = f"res_lr_{sgd_lr}.csv"
    save_path = os.path.join(output_dir, file_name)
    df.to_csv(save_path, index=False)
    print(f"Saved results for vanilla sgd lr={sgd_lr} to {save_path}")


def save_sgd_ewc_csv(sgd_lr, ewc_lambda, epochs_x, tr_m, tr_s, te_m, te_s, output_dir='results_cifar10_random/sgd_ewc_csv'):
    # 1. Create a DataFrame where each row is one epoch
    df = pd.DataFrame({
        'epoch': epochs_x,
        'sgd_lr': sgd_lr,
        'sgd_ewc': ewc_lambda,
        'train_acc_mean': tr_m,
        'train_acc_std': tr_s,
        'test_acc_mean': te_m,
        'test_acc_std': te_s
    })

    # 3. Save to CSV
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    file_name = f"res_lr_{sgd_lr}_ewc_{ewc_lambda}.csv"
    save_path = os.path.join(output_dir, file_name)
    df.to_csv(save_path, index=False)
    print(f"Saved results for lr={sgd_lr}, ewc_lambda={ewc_lambda} to {save_path}")

def plot_all_sgd_online_correction(results_dir="results"):
    all_files = glob.glob(os.path.join(results_dir, "res_eta_*.csv"))
    if not all_files:
        print("No CSV files found!")
        return

    # Combine all individual job results into one master DataFrame
    df = pd.concat((pd.read_csv(f) for f in all_files), ignore_index=True)

    fig, ax = plt.subplots(1, 2, figsize=(15, 6))

    # Get unique parameter pairs for the legend
    params = df.groupby(['sgd_eta', 'sgd_gamma'])

    for (eta, gamma), group in params:
        label = f"eta={eta}, gamma={gamma}"

        # Train Plot
        ax[0].plot(group['epoch'], group['train_acc_mean'], label=label)
        ax[0].set_ylim([0, 1])
        ax[0].fill_between(group['epoch'],
                           group['train_acc_mean'] - group['train_acc_std'],
                           group['train_acc_mean'] + group['train_acc_std'], alpha=0.2)

        # Test Plot
        ax[1].plot(group['epoch'], group['test_acc_mean'], label=label)
        ax[1].set_ylim([0, 1])
        ax[1].fill_between(group['epoch'],
                           group['test_acc_mean'] - group['test_acc_std'],
                           group['test_acc_mean'] + group['test_acc_std'], alpha=0.2)

    ax[0].set_title("Training Accuracy")
    ax[1].set_title("Testing Accuracy")
    for a in ax:
        a.set_xlabel("Epochs")
        a.set_ylabel("Accuracy")
        a.legend()

    plt.tight_layout()
    plt.show()


def plot_all_sgd_ewc(results_dir="results"):
    all_files = glob.glob(os.path.join(results_dir, "res_lr_*.csv"))
    if not all_files:
        print("No CSV files found!")
        return

    # Combine all individual job results into one master DataFrame
    df = pd.concat((pd.read_csv(f) for f in all_files), ignore_index=True)

    fig, ax = plt.subplots(1, 2, figsize=(15, 6))

    # Get unique parameter pairs for the legend
    params = df.groupby(['sgd_lr', 'sgd_ewc'])

    for (lr, ewc_lambda), group in params:
        label = f"lr={lr}, ewc_lambda={ewc_lambda}"

        # Train Plot
        ax[0].plot(group['epoch'], group['train_acc_mean'], label=label)
        ax[0].set_ylim([0, 0.3])
        ax[0].fill_between(group['epoch'],
                           group['train_acc_mean'] - group['train_acc_std'],
                           group['train_acc_mean'] + group['train_acc_std'], alpha=0.2)

        # Test Plot
        ax[1].plot(group['epoch'], group['test_acc_mean'], label=label)
        ax[1].set_ylim([0, 0.3])
        ax[1].fill_between(group['epoch'],
                           group['test_acc_mean'] - group['test_acc_std'],
                           group['test_acc_mean'] + group['test_acc_std'], alpha=0.2)

    ax[0].set_title("Training Accuracy")
    ax[1].set_title("Testing Accuracy")
    for a in ax:
        a.set_xlabel("Epochs")
        a.set_ylabel("Accuracy")
        a.legend()

    plt.tight_layout()
    plt.show()


def plot_experiment_comparison(results_dir, file_mapping, train_order_type='class_incremental', Ntrain=2000, criterion='MSE'):
    # 1. Define the mapping between filenames and your desired labels
    # This matches the filenames seen in your directory screenshot

    # Initialize figures based on your attached code format
    fig_tr, ax_tr = plt.subplots(figsize=(8, 6))
    fig_te, ax_te = plt.subplots(figsize=(8, 6))

    # 2. Process each file
    for filename, label in file_mapping.items():
        file_path = os.path.join(results_dir, filename)

        if not os.path.exists(file_path):
            print(f"Warning: {filename} not found in {results_dir}")
            continue

        df = pd.read_csv(file_path)

        # Plot Training Curve
        ax_tr.plot(df['epoch'], df['train_acc_mean'], label=label)
        ax_tr.fill_between(df['epoch'],
                           df['train_acc_mean'] - df['train_acc_std'],
                           df['train_acc_mean'] + df['train_acc_std'], alpha=0.2)

        # Plot Testing Curve
        ax_te.plot(df['epoch'], df['test_acc_mean'], label=label)
        ax_te.fill_between(df['epoch'],
                           df['test_acc_mean'] - df['test_acc_std'],
                           df['test_acc_mean'] + df['test_acc_std'], alpha=0.2)

    # 3. Apply your specific formatting from the code screenshots
    for ax, title in zip([ax_tr, ax_te], [f"Train Accuracy, Ntrain={Ntrain}, {train_order_type}, {criterion}", f"Test Accuracy, Ntrain={Ntrain}, {train_order_type}, {criterion}"]):
        ax.set_title(title)
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Accuracy")
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    fig_tr.tight_layout()
    fig_te.tight_layout()

    # 1. Define the full paths for saving
    # This places the .png files inside the 'results_dir' folder
    save_path_tr = os.path.join(results_dir, "train_comparison.png")
    save_path_te = os.path.join(results_dir, "test_comparison.png")

    # 2. Save using the full paths and your specific formatting
    fig_tr.savefig(save_path_tr, dpi=200, bbox_inches="tight")
    fig_te.savefig(save_path_te, dpi=200, bbox_inches="tight")

    plt.show()


def save_classification_csv(steps, sgd_tr_cor_m, sgd_tr_cor_s, sgd_te_cor_m, sgd_te_cor_s, sgd_eta, sgd_gamma, label_correction_type, hy_params, save_dir):
    # Pack the results into a dictionary, injecting the hyperparameters
    results_dict = {
        'step': steps,
        'train_acc_mean': sgd_tr_cor_m,
        'train_acc_sem': sgd_tr_cor_s,
        'test_acc_mean': sgd_te_cor_m,
        'test_acc_sem': sgd_te_cor_s,
        'sgd_eta': sgd_eta,
        'sgd_gamma': sgd_gamma,
        'label_correction_type': label_correction_type
    }

    # Convert to a DataFrame
    df = pd.DataFrame(results_dict)

    # If using a save directory, uncomment the line below:
    # save_path = os.path.join(save_dir, filename)
    os.makedirs(save_dir, exist_ok=True)

    filename = f"sgd_{label_correction_type}_eta{sgd_eta}_gm{sgd_gamma}_stat.csv"
    save_path = os.path.join(save_dir, filename)

    # Export to CSV without the pandas index column
    df.to_csv(save_path, index=False)
    print(f"Successfully saved tracking data to: {save_path}")

def save_correction_one_seed_csv(steps, acc_tr_cor, acc_te_cor, sgd_eta, sgd_gamma, label_correction_type, seed_id, save_dir):
    # Pack the results into a dictionary, injecting the hyperparameters
    results_dict = {
        'step': steps,
        'train_acc': acc_tr_cor,
        'test_acc': acc_te_cor,
        'sgd_eta': sgd_eta,
        'sgd_gamma': sgd_gamma,
        'seed_id': seed_id,
        'label_correction_type': label_correction_type
    }

    # Convert to a DataFrame
    df = pd.DataFrame(results_dict)

    # If using a save directory, uncomment the line below:
    # --- NEW: Create a specific subdirectory for this seed ---
    seed_dir = os.path.join(save_dir, str(seed_id))  # e.g., creates "save_dir/10"
    os.makedirs(seed_dir, exist_ok=True)

    filename = f"sgd_{label_correction_type}_eta{sgd_eta}_gm{sgd_gamma}_seed{seed_id}.csv"
    save_path = os.path.join(seed_dir, filename)

    # Export to CSV without the pandas index column
    df.to_csv(save_path, index=False)
    print(f"Successfully saved tracking data to: {save_path}")

def save_sgd_vanilla_one_seed_csv(steps, acc_tr, acc_te, sgd_lr, seed_id, save_dir='results'):
    # Pack the results into a dictionary, injecting the hyperparameters
    results_dict = {
        'step': steps,
        'train_acc': acc_tr,
        'test_acc': acc_te,
        'sgd_lr': sgd_lr,
        'seed_id': seed_id,
    }

    # Convert to a DataFrame
    df = pd.DataFrame(results_dict)

    seed_dir = os.path.join(save_dir, str(seed_id))  # e.g., creates "save_dir/10"
    os.makedirs(seed_dir, exist_ok=True)

    filename = f"sgd_van_lr{sgd_lr}_seed{seed_id}.csv"
    save_path = os.path.join(seed_dir, filename)

    # Export to CSV without the pandas index column
    df.to_csv(save_path, index=False)
    print(f"Successfully saved tracking data to: {save_path}")

def save_classification_vanilla_csv(steps, sgd_tr_cor_m, sgd_tr_cor_s, sgd_te_cor_m, sgd_te_cor_s, sgd_lr, hy_params, epochs=1, mini_bz=1, save_dir='results'):
    # Pack the results into a dictionary, injecting the hyperparameters
    results_dict = {
        'step': steps,
        'train_acc_mean': sgd_tr_cor_m,
        'train_acc_sem': sgd_tr_cor_s,
        'test_acc_mean': sgd_te_cor_m,
        'test_acc_sem': sgd_te_cor_s,
        'sgd_lr': sgd_lr,
        'epochs': epochs,
        'batch_size': mini_bz,
    }

    # Convert to a DataFrame
    df = pd.DataFrame(results_dict)

    os.makedirs(save_dir, exist_ok=True)

    filename = f"sgd_van_lr{sgd_lr}_epochs{epochs}_bz{mini_bz}.csv"
    save_path = os.path.join(save_dir, filename)

    # Export to CSV without the pandas index column
    df.to_csv(save_path, index=False)
    print(f"Successfully saved tracking data to: {save_path}")


def save_classification_cumulative_csv(steps, sgd_tr_cor_m, sgd_tr_cor_s, sgd_te_cor_m, sgd_te_cor_s, sgd_lr, hy_params, epochs=1, mini_bz=1, save_dir='results'):
    """Aggregated mean +/- SEM for task-incremental cumulative-replay baseline."""
    results_dict = {
        'step': steps,
        'train_acc_mean': sgd_tr_cor_m,
        'train_acc_sem': sgd_tr_cor_s,
        'test_acc_mean': sgd_te_cor_m,
        'test_acc_sem': sgd_te_cor_s,
        'sgd_lr': sgd_lr,
        'epochs': epochs,
        'batch_size': mini_bz,
        'label_correction_type': 'cumulative',
    }
    df = pd.DataFrame(results_dict)
    os.makedirs(save_dir, exist_ok=True)
    filename = f"sgd_cumulative_lr{sgd_lr}_epochs{epochs}_bz{mini_bz}.csv"
    save_path = os.path.join(save_dir, filename)
    df.to_csv(save_path, index=False)
    print(f"Successfully saved cumulative summary CSV to: {save_path}")


def save_sgd_cumulative_one_seed_csv(steps, acc_tr, acc_te, sgd_lr, seed_id, save_dir='results'):
    """Per-seed train/test accuracy curve for cumulative (same layout as vanilla one-seed CSV)."""
    results_dict = {
        'step': steps,
        'train_acc': acc_tr,
        'test_acc': acc_te,
        'sgd_lr': sgd_lr,
        'seed_id': seed_id,
        'label_correction_type': 'cumulative',
    }
    df = pd.DataFrame(results_dict)
    seed_dir = os.path.join(save_dir, str(seed_id))
    os.makedirs(seed_dir, exist_ok=True)
    filename = f"sgd_cumulative_lr{sgd_lr}_seed{seed_id}.csv"
    save_path = os.path.join(seed_dir, filename)
    df.to_csv(save_path, index=False)
    print(f"Successfully saved cumulative per-seed CSV to: {save_path}")


def save_classification_ewc_csv(steps, sgd_tr_cor_m, sgd_tr_cor_s, sgd_te_cor_m, sgd_te_cor_s, ewc_lambda, fixed_sgd_lr, hy_params, save_dir):
    # Pack the results into a dictionary, injecting the hyperparameters
    results_dict = {
        'step': steps,
        'train_acc_mean': sgd_tr_cor_m,
        'train_acc_sem': sgd_tr_cor_s,
        'test_acc_mean': sgd_te_cor_m,
        'test_acc_sem': sgd_te_cor_s,
        'ewc_lambda': ewc_lambda,
        'sgd_lr': fixed_sgd_lr,
    }

    # Convert to a DataFrame
    df = pd.DataFrame(results_dict)

    # If using a save directory, uncomment the line below:
    # save_path = os.path.join(save_dir, filename)
    os.makedirs(save_dir, exist_ok=True)

    filename = f"sgd_ewc_lamdba{ewc_lambda}_lr{fixed_sgd_lr}.csv"
    save_path = os.path.join(save_dir, filename)

    # Export to CSV without the pandas index column
    df.to_csv(save_path, index=False)
    print(f"Successfully saved tracking data to: {save_path}")

def save_classification_gating_csv(steps, sgd_tr_cor_m, sgd_tr_cor_s, sgd_te_cor_m, sgd_te_cor_s, gate_frac, fixed_sgd_lr, hy_params, gate_mode, save_dir):
    # Pack the results into a dictionary, injecting the hyperparameters
    results_dict = {
        'step': steps,
        'train_acc_mean': sgd_tr_cor_m,
        'train_acc_sem': sgd_tr_cor_s,
        'test_acc_mean': sgd_te_cor_m,
        'test_acc_sem': sgd_te_cor_s,
        'gate_frac': gate_frac,
        'sgd_lr': fixed_sgd_lr,
    }

    # Convert to a DataFrame
    df = pd.DataFrame(results_dict)
    os.makedirs(save_dir, exist_ok=True)

    filename = f"sgd_gate_{gate_mode}_frac{gate_frac}_lr{fixed_sgd_lr}.csv"
    save_path = os.path.join(save_dir, filename)

    # Export to CSV without the pandas index column
    df.to_csv(save_path, index=False)
    print(f"Successfully saved tracking data to: {save_path}")

def save_sgd_correction_csv(steps, sgd_tr_cor_m, sgd_tr_cor_s, sgd_te_cor_m, sgd_te_cor_s, eta, gamma, save_dir):
    # Pack the results into a dictionary, injecting the hyperparameters
    results_dict = {
        'step': steps,
        'train_acc_mean': sgd_tr_cor_m,
        'train_acc_sem': sgd_tr_cor_s,
        'test_acc_mean': sgd_te_cor_m,
        'test_acc_sem': sgd_te_cor_s,
        'eta': eta,
        'gamma': gamma,
    }

    # Convert to a DataFrame
    df = pd.DataFrame(results_dict)
    os.makedirs(save_dir, exist_ok=True)

    filename = f"sgd_online_correction_eta{eta}_gamma{gamma}.csv"
    save_path = os.path.join(save_dir, filename)

    df.to_csv(save_path, index=False)
    print(f"Successfully saved tracking data to: {save_path}")


def save_off_reg_csv(steps, off_tr_m, off_tr_s, off_te_m, off_te_s, gm, hy_params, save_dir):
    # Pack the results into a dictionary, injecting the hyperparameters
    results_dict = {
        'step': steps,
        'train_acc_mean': off_tr_m,
        'train_acc_sem': off_tr_s,
        'test_acc_mean': off_te_m,
        'test_acc_sem': off_te_s,
        'gamma': gm,
    }

    # Convert to a DataFrame
    df = pd.DataFrame(results_dict)
    os.makedirs(save_dir, exist_ok=True)

    filename = f"off_reg_N_tr{hy_params['Ntrain']}_gm{gm}.csv"
    save_path = os.path.join(save_dir, filename)
    df.to_csv(save_path, index=False)
    print(f"Successfully saved tracking data to: {save_path}")

import re
def plot_cil_strategy_comparison(results_dir, save_filename="cifar10_task_incre_comparison.png", data_tp='MNIST', strategy='Task-Incremental'):
    # Find all CSV files in the directory
    csv_pattern = os.path.join(results_dir, "*.csv")
    csv_files = glob.glob(csv_pattern)

    if not csv_files:
        print(f"No CSV files found in {results_dir}")
        return

    # Set up a professional figure style
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), sharey=True)


    # Helper function to map filenames to clean, LaTeX-formatted labels
    def get_label(filename):
        fname = os.path.basename(filename)

        if "sgd_van" in fname:
            # Looks for 'lr' followed by any numbers/decimals
            lr = re.search(r'lr([\d.]+)', fname)
            lr_val = lr.group(1) if lr else "unknown"

            return f"Vanilla SGD (lr={lr_val})"



        elif "ewc" in fname:
            lam = re.search(r'lamdba([\d.]+)', fname)
            lr = re.search(r'lr([\d.]+)', fname)
            lam_val = lam.group(1) if lam else "unknown"
            lr_val = lr.group(1) if lr else "unknown"
            if "old" in fname:
                return rf"EWC(old) ($\lambda={lam_val}$, lr={lr_val})"
            else:
                return rf"EWC(sample-wise) ($\lambda={lam_val}$, lr={lr_val})"

        elif "gate" in fname:
            frac = re.search(r'frac([\d.]+)', fname)
            lr = re.search(r'lr([\d.]+)', fname)
            frac_val = frac.group(1) if frac else "unknown"
            lr_val = lr.group(1) if lr else "unknown"

            return f"Gate oracle (frac={frac_val}, lr={lr_val})"

        elif "online_correction_update1" in fname:
            eta = re.search(r'eta([\d.]+)', fname)
            gamma = re.search(r'gm([\d.]+)', fname)
            eta_val = eta.group(1) if eta else "unknown"
            gamma_val = gamma.group(1) if gamma else "unknown"

            # Optional: You can also extract the 'fixedFirst' / 'noFixedFirst' status if needed
            status = " (First-Epoch Kernel)" if "fixedFirst" in fname else ""

            return rf"Online Correction Fixed Kernel (lr$ = \eta={eta_val}$, $\gamma={gamma_val}$){status}"

        elif "online_correction" in fname:
            eta = re.search(r'eta([\d.]+)', fname)
            gamma = re.search(r'gm([\d.]+)', fname)
            eta_val = eta.group(1) if eta else "unknown"
            gamma_val = gamma.group(1) if gamma else "unknown"

            # Optional: You can also extract the 'fixedFirst' / 'noFixedFirst' status if needed
            status = " (First-Epoch Kernel)" if "fixedFirst" in fname else ""

            return rf"Online Correction Updated Kernel (lr$ = \eta={eta_val}$, $\gamma={gamma_val}$){status}"



        return fname  # Fallback
    csv_files.sort()

    for file in csv_files:
        df = pd.read_csv(file)
        label = get_label(file)

        # Standardize columns based on your output format
        steps = df['step']
        # Calculate custom x-ticks based on your task boundaries
        total_steps = int(steps.max())  # dynamically gets 30000 from your dataframe
        num_tasks = 5
        step_gap = total_steps // num_tasks
        task_boundaries = range(0, total_steps + 1, step_gap)

        # --- Plot Training Accuracy ---
        train_mean = df['train_acc_mean']
        # Handle cases where the SEM column might be entirely zeros or missing
        train_sem = df['train_acc_sem'] if 'train_acc_sem' in df.columns else 0

        line, = ax1.plot(steps, train_mean, label=label, linewidth=2)
        if 'train_acc_sem' in df.columns:
            ax1.fill_between(steps, train_mean - train_sem, train_mean + train_sem,
                             alpha=0.2, color=line.get_color())

        # --- Plot Testing Accuracy ---
        test_mean = df['test_acc_mean']
        test_sem = df['test_acc_sem'] if 'test_acc_sem' in df.columns else 0

        ax2.plot(steps, test_mean, label=label, linewidth=2, color=line.get_color())
        if 'test_acc_sem' in df.columns:
            ax2.fill_between(steps, test_mean - test_sem, test_mean + test_sem,
                             alpha=0.2, color=line.get_color())

    # Format Training Subplot
    ax1.set_title("Training Accuracy", fontsize=14, fontweight='bold')
    ax1.set_xlabel("Streaming Steps", fontsize=12)
    ax1.set_ylabel("Accuracy", fontsize=12)
    ax1.set_ylim(0, 1.05)  # Locked 0 to 100% scale, with slight headroom
    ax1.set_xticks(task_boundaries)
    ax1.tick_params(axis='both', labelsize=11)
    ax1.legend(loc="lower left", fontsize=11, frameon=True)  # Lower left usually works best for CIL drops

    # Format Testing Subplot
    ax2.set_title("Testing Accuracy", fontsize=14, fontweight='bold')
    ax2.set_xlabel("Streaming Steps", fontsize=12)
    ax2.tick_params(axis='both', labelsize=11)
    ax2.set_xticks(task_boundaries)
    ax2.legend(loc="lower left", fontsize=11, frameon=True)

    # Final Polish and Save
    plt.suptitle(f"{data_tp} {strategy} Learning: Strategy Comparison",
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()

    # Save to the same folder as the CSVs
    save_path = os.path.join(results_dir, save_filename)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Plot successfully saved to: {save_path}")
    plt.show()

def plot_grid_search_results(target_dir, save_filename="continual_learning_plot.png"):
    # 1. Construct the search path using the target directory
    file_pattern = os.path.join(target_dir, "sgd_online_correction_*.csv")
    file_paths = glob.glob(file_pattern)

    if not file_paths:
        print(f"No CSV files found in directory: {target_dir}")
        return

    # 2. Read and sort the data so the legend looks clean
    data_list = []
    for file in file_paths:
        df = pd.read_csv(file)
        eta = df['sgd_eta'].iloc[0]
        gamma = df['sgd_gamma'].iloc[0]
        data_list.append((eta, gamma, df))

    # Sort numerically by eta first, then by gamma
    data_list.sort(key=lambda x: (x[0], x[1]))

    # 3. Set up the figure
    fig, (ax_train, ax_test) = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    # 4. Loop through the sorted data and plot
    for eta, gamma, df in data_list:
        plot_label = f"eta={eta}, gamma={gamma}"

        # Plot Training Accuracy
        ax_train.plot(df['step'], df['train_acc_mean'], label=plot_label, linewidth=2)
        ax_train.fill_between(df['step'],
                              df['train_acc_mean'] - df['train_acc_sem'],
                              df['train_acc_mean'] + df['train_acc_sem'],
                              alpha=0.2)

        # Plot Testing Accuracy
        ax_test.plot(df['step'], df['test_acc_mean'], label=plot_label, linewidth=2)
        ax_test.fill_between(df['step'],
                             df['test_acc_mean'] - df['test_acc_sem'],
                             df['test_acc_mean'] + df['test_acc_sem'],
                             alpha=0.2)

    # 5. Formatting
    # --- NEW: Lock the y-axis to [0, 1] ---
    ax_train.set_ylim(0, 1)

    ax_train.set_title("Training Accuracy", fontsize=14, fontweight='bold')
    ax_train.set_xlabel("Streaming Steps", fontsize=12)
    ax_train.set_ylabel("Accuracy", fontsize=12)
    ax_train.grid(True, linestyle='--', alpha=0.6)
    ax_train.legend(fontsize=10)

    ax_test.set_title("Testing Accuracy", fontsize=14, fontweight='bold')
    ax_test.set_xlabel("Streaming Steps", fontsize=12)
    ax_test.grid(True, linestyle='--', alpha=0.6)
    ax_test.legend(fontsize=10)

    plt.tight_layout()
    # 6. Save the plot to the target directory
    save_path = os.path.join(target_dir, save_filename)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Plot successfully saved to: {save_path}")

    # Display the plot if running interactively
    plt.show()

def plot_grid_search_results_one_seed(target_dir, save_filename="continual_learning_plot.png"):
    # 1. Construct the search path using the target directory
    file_pattern = os.path.join(target_dir, "sgd_online_correction_*.csv")
    file_paths = glob.glob(file_pattern)

    if not file_paths:
        print(f"No CSV files found in directory: {target_dir}")
        return

    # 2. Read and sort the data so the legend looks clean
    data_list = []
    for file in file_paths:
        df = pd.read_csv(file)
        eta = df['sgd_eta'].iloc[0]
        gamma = df['sgd_gamma'].iloc[0]
        data_list.append((eta, gamma, df))

    # Sort numerically by eta first, then by gamma
    data_list.sort(key=lambda x: (x[0], x[1]))

    # 3. Set up the figure
    fig, (ax_train, ax_test) = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    # 4. Loop through the sorted data and plot
    for eta, gamma, df in data_list:
        plot_label = f"eta={eta}, gamma={gamma}"

        # Plot Training Accuracy
        ax_train.plot(df['step'], df['train_acc'], label=plot_label, linewidth=2)


        # Plot Testing Accuracy
        ax_test.plot(df['step'], df['test_acc'], label=plot_label, linewidth=2)

    # 5. Formatting
    # --- NEW: Lock the y-axis to [0, 1] ---
    ax_train.set_ylim(0, 1)

    ax_train.set_title("Training Accuracy", fontsize=14, fontweight='bold')
    ax_train.set_xlabel("Streaming Steps", fontsize=12)
    ax_train.set_ylabel("Accuracy", fontsize=12)
    ax_train.grid(True, linestyle='--', alpha=0.6)
    ax_train.legend(fontsize=10)

    ax_test.set_title("Testing Accuracy", fontsize=14, fontweight='bold')
    ax_test.set_xlabel("Streaming Steps", fontsize=12)
    ax_test.grid(True, linestyle='--', alpha=0.6)
    ax_test.legend(fontsize=10)

    plt.tight_layout()
    # 6. Save the plot to the target directory
    save_path = os.path.join(target_dir, save_filename)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Plot successfully saved to: {save_path}")

    # Display the plot if running interactively
    plt.show()

def plot_krr_results(target_dir, save_filename="krr_learning_plot.png"):
    # 1. Construct the search path
    # This will catch files like "krr_results_N_tr10000_gamma_0.01.csv"
    file_pattern = os.path.join(target_dir, "krr_results_*gamma_*.csv")
    file_paths = glob.glob(file_pattern)

    if not file_paths:
        print(f"No CSV files found in directory: {target_dir}")
        return

    # 2. Read and sort the data
    data_list = []
    for file in file_paths:
        df = pd.read_csv(file)

        # Extract gamma directly from the column data
        gamma = df['gamma'].iloc[0]
        data_list.append((gamma, df))

    # Sort numerically by gamma from lowest to highest
    data_list.sort(key=lambda x: x[0])

    # 3. Set up the figure
    fig, (ax_train, ax_test) = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    # 4. Loop through the sorted data and plot
    for gamma, df in data_list:
        plot_label = f"gamma={gamma}"

        # --- Plot Training Accuracy ---
        ax_train.plot(df['step'], df['train_mean'], label=plot_label, linewidth=2)
        ax_train.fill_between(df['step'],
                              df['train_mean'] - df['train_sem'],
                              df['train_mean'] + df['train_sem'],
                              alpha=0.2)

        # --- Plot Testing Accuracy ---
        ax_test.plot(df['step'], df['test_mean'], label=plot_label, linewidth=2)
        ax_test.fill_between(df['step'],
                             df['test_mean'] - df['test_sem'],
                             df['test_mean'] + df['test_sem'],
                             alpha=0.2)

    # 5. Formatting
    ax_train.set_ylim(0, 1)  # Lock the y-axis strictly to [0, 1]

    ax_train.set_title("Training Accuracy (KRR)", fontsize=14, fontweight='bold')
    ax_train.set_xlabel("Streaming Steps", fontsize=12)
    ax_train.set_ylabel("Accuracy", fontsize=12)
    ax_train.grid(True, linestyle='--', alpha=0.6)
    ax_train.legend(fontsize=10)

    ax_test.set_title("Testing Accuracy (KRR)", fontsize=14, fontweight='bold')
    ax_test.set_xlabel("Streaming Steps", fontsize=12)
    ax_test.grid(True, linestyle='--', alpha=0.6)
    ax_test.legend(fontsize=10)

    plt.tight_layout()

    # 6. Save the plot to the target directory
    save_path = os.path.join(target_dir, save_filename)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Plot successfully saved to: {save_path}")

    # Display the plot if running interactively
    plt.show()


def plot_final_accuracy_vs_gamma(target_dir, target_step=30000, save_filename="gamma_vs_accuracy.png"):
    # 1. Construct the search path
    file_pattern = os.path.join(target_dir, "krr_results_N_tr30000_gamma_*.csv")
    file_paths = glob.glob(file_pattern)

    if not file_paths:
        print(f"No CSV files found matching {file_pattern}")
        return

    # 2. Extract the data for the specific step
    data_points = []
    for file in file_paths:
        df = pd.read_csv(file)

        # Filter the dataframe to only include the row where step == 30000
        step_df = df[df['step'] == target_step]

        if step_df.empty:
            print(f"Warning: Step {target_step} not found in {file}. Skipping.")
            continue

        gamma = step_df['gamma'].iloc[0]
        test_mean = step_df['test_mean'].iloc[0]
        test_sem = step_df['test_sem'].iloc[0]

        data_points.append((gamma, test_mean, test_sem))

    # 3. Sort strictly by gamma (lowest to highest) so the line connects properly
    data_points.sort(key=lambda x: x[0])

    # Unpack the sorted data into separate lists for plotting
    gammas = [item[0] for item in data_points]
    test_means = [item[1] for item in data_points]
    test_sems = [item[2] for item in data_points]

    # Convert gamma to log10(gamma)
    log_gammas = np.log10(gammas)

    # 4. Create the plot
    plt.figure(figsize=(8, 6))

    # errorbar automatically plots the line, the markers, and the vertical error bars
    plt.errorbar(log_gammas, test_means, yerr=test_sems,
                 fmt='-o',  # Line with circle markers
                 capsize=5,  # Adds caps to the error bars
                 linewidth=2,
                 markersize=8,
                 color='royalblue',
                 label=f'Test Accuracy at Step {target_step}')

    # 5. Formatting
    plt.title("Effect of Gamma on Final Test Accuracy (KRR)", fontsize=14, fontweight='bold')
    plt.xlabel(r"$\log_{10}(\gamma)$", fontsize=12)
    plt.ylabel("Test Accuracy", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=11)

    # Optional: If you want to force the y-axis to start from 0
    plt.ylim(0, 1)

    plt.tight_layout()

    # 6. Save and show
    save_path = os.path.join(target_dir, save_filename)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Plot successfully saved to: {save_path}")

    plt.show()


def plot_vanilla_sgd_results(target_dir, save_filename="vanilla_sgd_learning_plot.png"):
    # 1. Construct the search path
    file_pattern = os.path.join(target_dir, "sgd_van_*.csv")
    file_paths = glob.glob(file_pattern)

    if not file_paths:
        print(f"No CSV files found matching {file_pattern}")
        return

    # 2. Read and sort the data
    data_list = []
    for file in file_paths:
        df = pd.read_csv(file)

        # Extract the learning rate directly from the column data
        lr = df['sgd_lr'].iloc[0]
        data_list.append((lr, df))

    # Sort numerically by learning rate from lowest to highest
    data_list.sort(key=lambda x: x[0])

    # 3. Set up the figure
    fig, (ax_train, ax_test) = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    # 4. Loop through the sorted data and plot
    for lr, df in data_list:
        plot_label = f"lr={lr}"

        # --- Plot Training Accuracy ---
        ax_train.plot(df['step'], df['train_acc_mean'], label=plot_label, linewidth=2)
        ax_train.fill_between(df['step'],
                              df['train_acc_mean'] - df['train_acc_sem'],
                              df['train_acc_mean'] + df['train_acc_sem'],
                              alpha=0.2)

        # --- Plot Testing Accuracy ---
        ax_test.plot(df['step'], df['test_acc_mean'], label=plot_label, linewidth=2)
        ax_test.fill_between(df['step'],
                             df['test_acc_mean'] - df['test_acc_sem'],
                             df['test_acc_mean'] + df['test_acc_sem'],
                             alpha=0.2)

    # 5. Formatting
    ax_train.set_ylim(0, 1)  # Lock the y-axis strictly to [0, 1]

    ax_train.set_title("Training Accuracy (Vanilla SGD)", fontsize=14, fontweight='bold')
    ax_train.set_xlabel("Streaming Steps", fontsize=12)
    ax_train.set_ylabel("Accuracy", fontsize=12)
    ax_train.grid(True, linestyle='--', alpha=0.6)
    ax_train.legend(fontsize=10)

    ax_test.set_title("Testing Accuracy (Vanilla SGD)", fontsize=14, fontweight='bold')
    ax_test.set_xlabel("Streaming Steps", fontsize=12)
    ax_test.grid(True, linestyle='--', alpha=0.6)
    ax_test.legend(fontsize=10)

    plt.tight_layout()

    # 6. Save and show
    save_path = os.path.join(target_dir, save_filename)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Plot successfully saved to: {save_path}")

    # Display the plot if running interactively
    plt.show()


def plot_vanilla_sgd_results_epochs(target_dir, save_filename="vanilla_sgd_learning_plot.png", epochs=3):
    # 1. Construct the search path
    file_pattern = os.path.join(target_dir, "sgd_van_*.csv")
    file_paths = glob.glob(file_pattern)

    if not file_paths:
        print(f"No CSV files found matching {file_pattern}")
        return

    # 2. Read and sort the data
    data_list = []
    for file in file_paths:
        df = pd.read_csv(file)

        # Extract the learning rate directly from the column data
        lr = df['sgd_lr'].iloc[0]
        data_list.append((lr, df))

    # Sort numerically by learning rate from lowest to highest
    data_list.sort(key=lambda x: x[0])

    # 3. Set up the figure
    fig, (ax_train, ax_test) = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    # 4. Loop through the sorted data and plot
    for lr, df in data_list:
        plot_label = f"lr={lr}"

        # --- Plot Training Accuracy ---
        ax_train.plot(df['step'], df['train_acc_mean'], label=plot_label, linewidth=2)
        ax_train.fill_between(df['step'],
                              df['train_acc_mean'] - df['train_acc_sem'],
                              df['train_acc_mean'] + df['train_acc_sem'],
                              alpha=0.2)

        # --- Plot Testing Accuracy ---
        ax_test.plot(df['step'], df['test_acc_mean'], label=plot_label, linewidth=2)
        ax_test.fill_between(df['step'],
                             df['test_acc_mean'] - df['test_acc_sem'],
                             df['test_acc_mean'] + df['test_acc_sem'],
                             alpha=0.2)

    # 5. Formatting
    total_epochs = int(epochs*5+1)
    ax_train.set_ylim(0, 1)  # Lock the y-axis strictly to [0, 1]
    ax_train.set_xticks(range(epochs, total_epochs, epochs))

    ax_train.set_title("Training Accuracy (Vanilla SGD)", fontsize=14, fontweight='bold')
    ax_train.set_xlabel("Epochs", fontsize=12)
    ax_train.set_ylabel("Accuracy", fontsize=12)
    ax_train.grid(True, linestyle='--', alpha=0.6)
    ax_train.legend(fontsize=10)
    ax_test.set_xticks(range(epochs, total_epochs, epochs))

    ax_test.set_title("Testing Accuracy (Vanilla SGD)", fontsize=14, fontweight='bold')
    ax_test.set_xlabel("Epochs", fontsize=12)
    ax_test.grid(True, linestyle='--', alpha=0.6)
    ax_test.legend(fontsize=10)

    plt.tight_layout()

    # 6. Save and show
    save_path = os.path.join(target_dir, save_filename)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Plot successfully saved to: {save_path}")

    # Display the plot if running interactively
    plt.show()

def plot_vanilla_sgd_results_one_seed(target_dir, save_filename="vanilla_sgd_learning_plot.png"):
    # 1. Construct the search path
    file_pattern = os.path.join(target_dir, "sgd_van_*.csv")
    file_paths = glob.glob(file_pattern)

    if not file_paths:
        print(f"No CSV files found matching {file_pattern}")
        return

    # 2. Read and sort the data
    data_list = []
    for file in file_paths:
        df = pd.read_csv(file)

        # Extract the learning rate directly from the column data
        lr = df['sgd_lr'].iloc[0]
        data_list.append((lr, df))

    # Sort numerically by learning rate from lowest to highest
    data_list.sort(key=lambda x: x[0])

    # 3. Set up the figure
    fig, (ax_train, ax_test) = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    # 4. Loop through the sorted data and plot
    for lr, df in data_list:
        plot_label = f"lr={lr}"

        # --- Plot Training Accuracy ---
        ax_train.plot(df['step'], df['train_acc'], label=plot_label, linewidth=2)
        # --- Plot Testing Accuracy ---
        ax_test.plot(df['step'], df['test_acc'], label=plot_label, linewidth=2)

    # 5. Formatting
    ax_train.set_ylim(0, 1)  # Lock the y-axis strictly to [0, 1]

    ax_train.set_title("Training Accuracy (Vanilla SGD)", fontsize=14, fontweight='bold')
    ax_train.set_xlabel("Streaming Steps", fontsize=12)
    ax_train.set_ylabel("Accuracy", fontsize=12)
    ax_train.grid(True, linestyle='--', alpha=0.6)
    ax_train.legend(fontsize=10)

    ax_test.set_title("Testing Accuracy (Vanilla SGD)", fontsize=14, fontweight='bold')
    ax_test.set_xlabel("Streaming Steps", fontsize=12)
    ax_test.grid(True, linestyle='--', alpha=0.6)
    ax_test.legend(fontsize=10)

    plt.tight_layout()

    # 6. Save and show
    save_path = os.path.join(target_dir, save_filename)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Plot successfully saved to: {save_path}")

    # Display the plot if running interactively
    plt.show()