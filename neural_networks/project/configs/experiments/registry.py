import fiddle as fdl

from .cnn_sweep import cnn_optimizer_sweep
from .lenet_sweep import lenet_optimizer_sweep
from .mlp_sweep import mlp_optimizer_sweep


def all_experiments() -> list[fdl.Config]:
    return [
        *mlp_optimizer_sweep(),
        *lenet_optimizer_sweep(),
        *cnn_optimizer_sweep(),
    ]
