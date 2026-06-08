import fiddle as fdl

from configs.constants import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_DEVICE,
    DEFAULT_DROPOUT,
    DEFAULT_EPOCHS,
    DEFAULT_HIDDEN_SIZE,
    DEFAULT_NUM_WORKERS,
    DEFAULT_SEED,
    INPUT_SIZE,
    NUM_CLASSES,
    TEST_CSV,
    TRAIN_CSV,
)

from .common import DatasetConfig, ExperimentConfig, ModelConfig, OptimizerConfig


def _dataset(batch_size: int = DEFAULT_BATCH_SIZE) -> fdl.Config:
    return fdl.Config(
        DatasetConfig,
        train_csv=TRAIN_CSV,
        test_csv=TEST_CSV,
        batch_size=batch_size,
        num_workers=DEFAULT_NUM_WORKERS,
        shuffle_train=True,
    )


def _cnn_experiment(
    name: str,
    optimizer_name: str,
    learning_rate: float,
    weight_decay: float = 0.0,
    momentum: float = 0.0,
) -> fdl.Config:
    return fdl.Config(
        ExperimentConfig,
        name=name,
        dataset=_dataset(batch_size=DEFAULT_BATCH_SIZE),
        model=fdl.Config(
            ModelConfig,
            name="cnn",
            input_size=INPUT_SIZE,
            hidden_size=DEFAULT_HIDDEN_SIZE,
            num_classes=NUM_CLASSES,
            dropout=DEFAULT_DROPOUT,
        ),
        optimizer=fdl.Config(
            OptimizerConfig,
            name=optimizer_name,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            momentum=momentum,
        ),
        epochs=DEFAULT_EPOCHS,
        seed=DEFAULT_SEED,
        device=DEFAULT_DEVICE,
    )


def cnn_adam_lr_1e_3() -> fdl.Config:
    return _cnn_experiment(
        name="cnn_adam_lr_1e_3",
        optimizer_name="adam",
        learning_rate=1e-3,
    )


def cnn_adam_lr_3e_4() -> fdl.Config:
    return _cnn_experiment(
        name="cnn_adam_lr_3e_4",
        optimizer_name="adam",
        learning_rate=3e-4,
    )


def cnn_sgd_lr_1e_2() -> fdl.Config:
    return _cnn_experiment(
        name="cnn_sgd_lr_1e_2",
        optimizer_name="sgd",
        learning_rate=1e-2,
    )


def cnn_sgd_momentum_lr_1e_2() -> fdl.Config:
    return _cnn_experiment(
        name="cnn_sgd_momentum_lr_1e_2",
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
