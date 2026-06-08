import fiddle as fdl

from .builders import experiment_config


def mlp_adam_lr_1e_3() -> fdl.Config:
    return experiment_config(
        name="mlp_adam_lr_1e_3",
        model_name="mlp",
        optimizer_name="adam",
        learning_rate=1e-3,
        dropout=0.0,
    )


def mlp_adam_lr_3e_4() -> fdl.Config:
    return experiment_config(
        name="mlp_adam_lr_3e_4",
        model_name="mlp",
        optimizer_name="adam",
        learning_rate=3e-4,
        dropout=0.0,
    )


def mlp_sgd_lr_1e_2() -> fdl.Config:
    return experiment_config(
        name="mlp_sgd_lr_1e_2",
        model_name="mlp",
        optimizer_name="sgd",
        learning_rate=1e-2,
        dropout=0.0,
    )


def mlp_sgd_momentum_lr_1e_2() -> fdl.Config:
    return experiment_config(
        name="mlp_sgd_momentum_lr_1e_2",
        model_name="mlp",
        optimizer_name="sgd",
        learning_rate=1e-2,
        momentum=0.9,
        dropout=0.0,
    )


def mlp_optimizer_sweep() -> list[fdl.Config]:
    return [
        mlp_adam_lr_1e_3(),
        mlp_adam_lr_3e_4(),
        mlp_sgd_lr_1e_2(),
        mlp_sgd_momentum_lr_1e_2(),
    ]
