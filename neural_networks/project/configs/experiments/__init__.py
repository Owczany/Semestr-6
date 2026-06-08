"""Experiment configs built with Google Fiddle."""

from .baselines import cnn_baseline, mlp_baseline
from .builders import dataset_config, experiment_config
from .cnn_sweep import (
    cnn_adam_lr_1e_3,
    cnn_adam_lr_3e_4,
    cnn_optimizer_sweep,
    cnn_sgd_lr_1e_2,
    cnn_sgd_momentum_lr_1e_2,
)
from .lenet_sweep import (
    lenet_adam_lr_1e_3,
    lenet_adam_lr_3e_4,
    lenet_optimizer_sweep,
    lenet_sgd_lr_1e_2,
    lenet_sgd_momentum_lr_1e_2,
)
from .mlp_sweep import (
    mlp_adam_lr_1e_3,
    mlp_adam_lr_3e_4,
    mlp_optimizer_sweep,
    mlp_sgd_lr_1e_2,
    mlp_sgd_momentum_lr_1e_2,
)
from .registry import all_experiments

__all__ = [
    "all_experiments",
    "cnn_adam_lr_1e_3",
    "cnn_adam_lr_3e_4",
    "cnn_baseline",
    "cnn_optimizer_sweep",
    "cnn_sgd_lr_1e_2",
    "cnn_sgd_momentum_lr_1e_2",
    "dataset_config",
    "experiment_config",
    "lenet_adam_lr_1e_3",
    "lenet_adam_lr_3e_4",
    "lenet_optimizer_sweep",
    "lenet_sgd_lr_1e_2",
    "lenet_sgd_momentum_lr_1e_2",
    "mlp_adam_lr_1e_3",
    "mlp_adam_lr_3e_4",
    "mlp_baseline",
    "mlp_optimizer_sweep",
    "mlp_sgd_lr_1e_2",
    "mlp_sgd_momentum_lr_1e_2",
]
