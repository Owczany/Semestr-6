import fiddle as fdl

from .builders import experiment_config


def cnn_adam_lr_1e_3() -> fdl.Config:
    return experiment_config(
        name="cnn_adam_lr_1e_3",
        model_name="cnn",
        optimizer_name="adam",
        learning_rate=1e-3,
    )


def cnn_adam_lr_3e_4() -> fdl.Config:
    return experiment_config(
        name="cnn_adam_lr_3e_4",
        model_name="cnn",
        optimizer_name="adam",
        learning_rate=3e-4,
    )


def cnn_sgd_lr_1e_2() -> fdl.Config:
    return experiment_config(
        name="cnn_sgd_lr_1e_2",
        model_name="cnn",
        optimizer_name="sgd",
        learning_rate=1e-2,
    )


def cnn_sgd_momentum_lr_1e_2() -> fdl.Config:
    return experiment_config(
        name="cnn_sgd_momentum_lr_1e_2",
        model_name="cnn",
        optimizer_name="sgd",
        learning_rate=1e-2,
        momentum=0.9,
    )


def cnn_optimizer_sweep() -> list[fdl.Config]:
    return [
        cnn_adam_lr_1e_3(),
        cnn_adam_lr_3e_4(),
        cnn_sgd_lr_1e_2(),
        cnn_sgd_momentum_lr_1e_2(),
    ]
