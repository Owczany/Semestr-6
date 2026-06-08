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


def dataset_config(batch_size: int = DEFAULT_BATCH_SIZE) -> fdl.Config:
    return fdl.Config(
        DatasetConfig,
        train_csv=TRAIN_CSV,
        test_csv=TEST_CSV,
        batch_size=batch_size,
        num_workers=DEFAULT_NUM_WORKERS,
        shuffle_train=True,
    )


def experiment_config(
    name: str,
    model_name: str,
    optimizer_name: str,
    learning_rate: float,
    weight_decay: float = 0.0,
    momentum: float = 0.0,
    batch_size: int = DEFAULT_BATCH_SIZE,
    hidden_size: int = DEFAULT_HIDDEN_SIZE,
    dropout: float = DEFAULT_DROPOUT,
    epochs: int = DEFAULT_EPOCHS,
    seed: int = DEFAULT_SEED,
    device: str = DEFAULT_DEVICE,
) -> fdl.Config:
    return fdl.Config(
        ExperimentConfig,
        name=name,
        dataset=dataset_config(batch_size=batch_size),
        model=fdl.Config(
            ModelConfig,
            name=model_name,
            input_size=INPUT_SIZE,
            hidden_size=hidden_size,
            num_classes=NUM_CLASSES,
            dropout=dropout,
        ),
        optimizer=fdl.Config(
            OptimizerConfig,
            name=optimizer_name,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            momentum=momentum,
        ),
        epochs=epochs,
        seed=seed,
        device=device,
    )
