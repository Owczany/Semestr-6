import fiddle as fdl

from .builders import experiment_config


def lenet_adam_lr_1e_3() -> fdl.Config:
    return experiment_config(
        name="lenet_adam_lr_1e_3",
        model_name="lenet",
        optimizer_name="adam",
        learning_rate=1e-3,
        dropout=0.0,
    )


def lenet_adam_lr_3e_4() -> fdl.Config:
    return experiment_config(
        name="lenet_adam_lr_3e_4",
        model_name="lenet",
        optimizer_name="adam",
        learning_rate=3e-4,
        dropout=0.0,
    )


def lenet_sgd_lr_1e_2() -> fdl.Config:
    return experiment_config(
        name="lenet_sgd_lr_1e_2",
        model_name="lenet",
        optimizer_name="sgd",
        learning_rate=1e-2,
        dropout=0.0,
    )


def lenet_sgd_momentum_lr_1e_2() -> fdl.Config:
    return experiment_config(
        name="lenet_sgd_momentum_lr_1e_2",
        model_name="lenet",
        optimizer_name="sgd",
        learning_rate=1e-2,
        momentum=0.9,
        dropout=0.0,
    )


def lenet_optimizer_sweep() -> list[fdl.Config]:
    return [
        lenet_adam_lr_1e_3(),
        lenet_adam_lr_3e_4(),
        lenet_sgd_lr_1e_2(),
        lenet_sgd_momentum_lr_1e_2(),
    ]
