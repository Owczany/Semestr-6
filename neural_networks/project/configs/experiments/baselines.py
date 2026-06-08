import fiddle as fdl

from .common import DatasetConfig, ExperimentConfig, ModelConfig, OptimizerConfig


def _sign_mnist_dataset(batch_size: int = 64) -> fdl.Config:
    return fdl.Config(
        DatasetConfig,
        train_csv="data/asl_mnist/sign_mnist_train.csv",
        test_csv="data/asl_mnist/sign_mnist_test.csv",
        batch_size=batch_size,
        num_workers=0,
        shuffle_train=True,
    )


def mlp_baseline() -> fdl.Config:
    return fdl.Config(
        ExperimentConfig,
        name="mlp_baseline",
        dataset=_sign_mnist_dataset(batch_size=64),
        model=fdl.Config(
            ModelConfig,
            name="mlp",
            input_size=28 * 28,
            hidden_size=256,
            num_classes=24,
            dropout=0.0,
        ),
        optimizer=fdl.Config(
            OptimizerConfig,
            name="adam",
            learning_rate=1e-3,
            weight_decay=0.0,
        ),
        epochs=4,
        seed=42,
        device="auto",
    )


def cnn_baseline() -> fdl.Config:
    return fdl.Config(
        ExperimentConfig,
        name="cnn_baseline",
        dataset=_sign_mnist_dataset(batch_size=64),
        model=fdl.Config(
            ModelConfig,
            name="cnn",
            input_size=28 * 28,
            hidden_size=256,
            num_classes=24,
            dropout=0.5,
        ),
        optimizer=fdl.Config(
            OptimizerConfig,
            name="adam",
            learning_rate=1e-3,
            weight_decay=0.0,
        ),
        epochs=4,
        seed=42,
        device="auto",
    )
