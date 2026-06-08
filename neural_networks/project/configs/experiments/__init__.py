"""Experiment configs built with Google Fiddle."""

from .baselines import cnn_baseline, mlp_baseline
from .cnn_sweep import (
    cnn_adam_lr_1e_3,
    cnn_adam_lr_3e_4,
    cnn_optimizer_sweep,
    cnn_sgd_lr_1e_2,
    cnn_sgd_momentum_lr_1e_2,
)

__all__ = [
    "cnn_adam_lr_1e_3",
    "cnn_adam_lr_3e_4",
    "cnn_baseline",
    "cnn_optimizer_sweep",
    "cnn_sgd_lr_1e_2",
    "cnn_sgd_momentum_lr_1e_2",
    "mlp_baseline",
]
