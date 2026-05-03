Code associated with the project on the effective target shift in online learning and its correction

- - -

## Toy Model

Simple toy model for illustration purpose

- - - 

## Image Classification

This repository contains two command-line entry scripts and a small `src` library for streaming SGD, target correction, EWC, and result plotting on **MNIST**, **CIFAR-10**, and **CORe50**-style pipelines.

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

From the repository root (same directory as the `main_*.py` files):

```bash
python main_cifar10_mnist.py --help
python main_core50.py --help
```

Typical knobs include dataset (`--ds_type`), stream order (`--train_order_type`), target correction mode (`--label_correction_type` / `--label_correction`), learning rates, batch sizes, and seeds (`--ikmax`). Exact flags differ slightly between the two mains; use `--help` for the authoritative list.

## Data

- **MNIST / CIFAR-10**: torchvision will download under `./data` (or your chosen `--root`) when missing; on clusters without reliable outbound HTTP, pre-stage the dataset there.
- **CORe50**: configure paths and preprocessing as required by `src/dataset.py` for your environment.

- - - 

## System Requirements

- - -
