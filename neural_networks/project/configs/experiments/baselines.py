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


def _sign_mnist_dataset(batch_size: int = DEFAULT_BATCH_SIZE) -> fdl.Config:
    return fdl.Config(
        DatasetConfig,
        train_csv=TRAIN_CSV,
        test_csv=TEST_CSV,
        batch_size=batch_size,
        num_workers=DEFAULT_NUM_WORKERS,
        shuffle_train=True,
    )


def mlp_baseline() -> fdl.Config:
    return fdl.Config(
        ExperimentConfig,
        name="mlp_baseline",
        dataset=_sign_mnist_dataset(batch_size=DEFAULT_BATCH_SIZE),
        model=fdl.Config(
            ModelConfig,
            name="mlp",
            input_size=INPUT_SIZE,
            hidden_size=DEFAULT_HIDDEN_SIZE,
            num_classes=NUM_CLASSES,
            dropout=0.0,
        ),
        optimizer=fdl.Config(
            OptimizerConfig,
            name="adam",
            learning_rate=1e-3,
            weight_decay=0.0,
        ),
        epochs=DEFAULT_EPOCHS,
        seed=DEFAULT_SEED,
        device=DEFAULT_DEVICE,
    )


def cnn_baseline() -> fdl.Config:
    return fdl.Config(
        ExperimentConfig,
        name="cnn_baseline",
        dataset=_sign_mnist_dataset(batch_size=DEFAULT_BATCH_SIZE),
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
            name="adam",
            learning_rate=1e-3,
            weight_decay=0.0,
        ),
        epochs=DEFAULT_EPOCHS,
        seed=DEFAULT_SEED,
        device=DEFAULT_DEVICE,
    )
