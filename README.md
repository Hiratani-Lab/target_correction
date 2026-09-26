Code associated with the project on the effective target shift in online learning and its correction

- - -

## Toy Model

Simple toy model for illustrative purposes

- - - 

## Conv-NTK Model

Illustration of effective label shifts and their full and iterative corrections using Neural Tangent Kernel (NTK) of a shallow convolutional neural network, applied to the MNIST dataset. 

NTK was estimated using the [neural_tangents library](https://github.com/google/neural-tangents) and implemented with JAX. 

- - - 

## Image Classification

This directory contains two command-line entry scripts and a small `src` library for streaming SGD, target correction, EWC, and result plotting on **MNIST**, **CIFAR-10**, and **CORe50**-style pipelines.

## Layout

| Path | Role |
|------|------|
| **`main_cifar10_mnist.py`** | Primary driver for **MNIST / CIFAR-10**: streaming training, target correction modes, vanilla / EWC / cumulative replay sweeps, CSV exports. |
| **`main_core50.py`** | Driver for **CORe50** subset experiments: vanilla SGD, cumulative replay, SGD+EWC, and target correction backends (`full_efficient`) with ResNet-18-style models. |
| **`src/`** | Shared implementation modules (imported via `sys.path` to `src/` next to each main script). |

## `src/` modules

| File | Contents |
|------|------------------------|
| **`dataset.py`** | `load_ds` / `load_cifar10_subset` / MNIST loaders; `make_order_multiclass` for random, class-incremental, and task-incremental orderings; CORe50 whitening loader used by `main_core50.py`. |
| **`nn_model.py`** | CNN definitions, streaming SGD with optional gating, **online target correction** (kernel + corrected targets `Z`), EWC hooks, CORe50 training loops, and routing to kernel backends via `kernel_type`. |
| **`kernel_cal.py`** | Empirical NTK / Jacobian-feature kernels, block-wise kernel assembly, helpers used by `nn_model` for correction and diagnostics. |
| **`ewc_gating.py`** | Diagonal Fisher and EWC penalty utilities used in streaming / gating setups. |
| **`plot.py`** | Save and plot learning curves and sweeps (e.g. target correction, vanilla SGD, EWC, cumulative-replay CSVs). |

## Dependencies

- **Python 3** with **PyTorch** and **torchvision** (versions should match your CUDA / CPU setup).
- **NumPy**, **pandas**, **matplotlib** for numerics, CSV I/O, and figures.
- CORe50 path may need local data layout expected by `dataset.load_core50_subset_whitened` (see that function for paths and options).

## Running

From the directory root (same directory as the `main_*.py` files):

```bash
python main_cifar10_mnist.py --help
python main_core50.py --help
```

Typical knobs include dataset (`--ds_type`), stream order (`--train_order_type`), target correction mode (`--label_correction_type` / `--label_correction`), learning rates, batch sizes, and seeds (`--ikmax`). Exact flags differ slightly between the two mains; use `--help` for the authoritative list.

## Replay and soft-label baselines

These runs write a mean/SEM CSV and do not create figures. One command is one configuration. To sweep, pass several values to a `*_list` flag, or launch one process per configuration.

CIFAR-10 or MNIST, one pass (`--epochs 1`). `label_smooth`, `ER`, `DER`, and `DER++` share the streaming SGD loop. Cumulative replay is the existing `cumulative` mode; `--replay_schedule task_boundary` replays at the end of each task, and `every_chunk` replays after every `--record_step`.

```bash
python main_cifar10_mnist.py \
  --ds_type cifar10 \
  --train_order_type task_incremental \
  --label_correction_type ER \
  --epochs 1 --ikmax 10 \
  --Ntrain 30000 --Ntest 10000 \
  --sgd_batch_size 4 --record_step 600 \
  --sgd_lr_list 0.01 \
  --der_buffer_size_list 300 \
  --der_replay_batch_size_list 4 \
  --der_alpha_list 1.0
```

Replace `ER` with `DER` or `DER++`. `DER++` also uses `--der_beta_list`. For soft labels:

```bash
python main_cifar10_mnist.py \
  --ds_type cifar10 \
  --train_order_type task_incremental \
  --label_correction_type label_smooth \
  --epochs 1 --ikmax 10 \
  --Ntrain 30000 --Ntest 10000 \
  --sgd_batch_size 4 --record_step 600 \
  --sgd_lr_list 0.01 \
  --label_smoothing_epsilon_list 0.1
```

The target is `(1 - epsilon) * one_hot + epsilon / C`, with `C = 2` for task-incremental binary tasks and `C = 10` otherwise. Training uses MSE. Accuracy is still computed against the original hard labels.

CORe50 uses the same method names on `--label_correction`. Each task is trained for `--epochs` passes over all samples seen so far for `cumulative`, or over the current task plus a reservoir for `ER`, `DER`, and `DER++`. Place `core50_imgs.npz` and `paths.pkl` in the working directory.

```bash
python main_core50.py \
  --label_correction DER \
  --sgd_lr 0.01 \
  --epochs 10 --ikmax 5 \
  --sgd_batch_size 4 --frames_per_session 15 \
  --der_buffer_size 600 \
  --der_replay_batch_size 4 \
  --der_alpha 0.01
```

`DER++` adds `--der_beta`. Soft labels use `--label_correction label_smooth` and `--label_smoothing_epsilon`. Here `C = 50`.

CSV files are written under `results_.../<method>/` for CIFAR-10 and MNIST, and under `results/core50_.../<method>/` for CORe50.

## Data

- **MNIST / CIFAR-10**: torchvision will download under `./data` (or your chosen `--root`) when missing; on clusters without reliable outbound HTTP, pre-stage the dataset there.
- **CORe50**: configure paths and preprocessing as required by `src/dataset.py` for your environment.


