import fiddle as fdl

from .builders import experiment_config


def mlp_baseline() -> fdl.Config:
    return experiment_config(
        name="mlp_baseline",
        model_name="mlp",
        optimizer_name="adam",
        learning_rate=1e-3,
        dropout=0.0,
    )


def cnn_baseline() -> fdl.Config:
    return experiment_config(
        name="cnn_baseline",
        model_name="cnn",
        optimizer_name="adam",
        learning_rate=1e-3,
    )
