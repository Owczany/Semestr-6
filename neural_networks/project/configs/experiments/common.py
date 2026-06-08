from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class DatasetConfig:
    train_csv: str
    test_csv: str
    batch_size: int
    num_workers: int
    shuffle_train: bool = True


@dataclass(frozen=True)
class ModelConfig:
    name: Literal["mlp", "cnn"]
    input_size: int = 28 * 28
    hidden_size: int = 256
    num_classes: int = 24
    dropout: float = 0.5


@dataclass(frozen=True)
class OptimizerConfig:
    name: Literal["adam"] = "adam"
    learning_rate: float = 1e-3
    weight_decay: float = 0.0


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    dataset: DatasetConfig
    model: ModelConfig
    optimizer: OptimizerConfig
    epochs: int
    seed: int = 42
    device: Literal["auto", "cpu", "cuda", "mps"] = "auto"
