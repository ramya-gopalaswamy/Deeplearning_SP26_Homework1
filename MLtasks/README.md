# ML Tasks (PyTorch)

Four new deep learning tasks following the **pytorch_task_v1** protocol (CoderGym-style). Each task is a single self-contained `task.py` implemented in PyTorch and self-verifiable via exit status.

## Tasks

| Task ID | Series | Description |
|--------|--------|-------------|
| `linreg_lvl5_sgd_scheduler` | Linear Regression | Univariate linear regression with mini-batch SGD, StepLR scheduler, and gradient clipping on synthetic data. |
| `logreg_lvl5_weight_decay_augment` | Logistic Regression | Binary classification on 2D Gaussian blobs with L2 weight decay and Gaussian noise augmentation. |
| `cnn_lvl5_augmented_mnist_like` | CNN | Small CNN on synthetic 16×16 blob images (3 classes) with data augmentation and early stopping. |
| `rnn_lvl5_gru_time_series` | Sequence Models | GRU-based sequence-to-one regression on synthetic noisy sine-wave time series. |

## Requirements

- **Python 3.x** (required; code is not compatible with Python 2). If not installed: [python.org/downloads](https://www.python.org/downloads/) or use your system package manager (e.g. `brew install python3`, `apt install python3`).
- **PyTorch** (`torch`)

Install PyTorch if needed:

```bash
pip install torch
```

## How to Run

From the `MLtasks` directory, use a **Python 3** interpreter. Depending on the system, that may be `python3` or `python`:

```bash
# Use whichever command runs Python 3 on your system
python3 tasks/<task_id>/task.py
# or
python tasks/<task_id>/task.py
```

Examples:

```bash
python3 tasks/linreg_lvl5_sgd_scheduler/task.py
python3 tasks/logreg_lvl5_weight_decay_augment/task.py
python3 tasks/cnn_lvl5_augmented_mnist_like/task.py
python3 tasks/rnn_lvl5_gru_time_series/task.py
```

If the evaluator’s system has Python 3 as `python` (e.g. many Linux/macOS setups), the same commands work with `python` instead of `python3`.

## Verification (Exit Status)

Each script is self-verifiable:

- **Exit 0** — Metrics meet the required thresholds (success).
- **Exit 1** — At least one threshold failed.

Check the exit code after running:

```bash
python3 tasks/linreg_lvl5_sgd_scheduler/task.py
echo "Exit code: $?"
```

Run all four tasks:

```bash
for t in linreg_lvl5_sgd_scheduler logreg_lvl5_weight_decay_augment cnn_lvl5_augmented_mnist_like rnn_lvl5_gru_time_series; do
  python3 tasks/$t/task.py
  echo "Exit code: $?"
done
```

## Success Criteria (Summary)

| Task | Pass condition |
|------|----------------|
| linreg_lvl5_sgd_scheduler | Val R² > 0.88, param L2 error < 0.5 |
| logreg_lvl5_weight_decay_augment | Val accuracy > 0.9, Val F1 > 0.9 |
| cnn_lvl5_augmented_mnist_like | Val accuracy > 0.9 |
| rnn_lvl5_gru_time_series | Val R² > 0.9, Val MSE < 0.05 |

## Structure

```
MLtasks/
├── README.md           # This file
├── ml_tasks.json       # Task definitions and pytorch_task_v1 protocol
└── tasks/
    ├── linreg_lvl5_sgd_scheduler/
    │   └── task.py
    ├── logreg_lvl5_weight_decay_augment/
    │   └── task.py
    ├── cnn_lvl5_augmented_mnist_like/
    │   └── task.py
    └── rnn_lvl5_gru_time_series/
        └── task.py
```

Artifacts (saved metrics/checkpoints) are written to `artifacts/` when each task runs.

## Reference

Tasks follow the protocol from [CoderGym MLtasks](https://github.com/lkk688/CoderGym/tree/main/MLtasks).
