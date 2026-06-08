import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.datasets import SignLanguageMNIST
from src.models import CNN, MLP


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device: str):
    if device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def make_data_loaders(dataset_config):
    train_dataset = SignLanguageMNIST(dataset_config.train_csv)
    test_dataset = SignLanguageMNIST(dataset_config.test_csv)

    train_loader = DataLoader(
        train_dataset,
        batch_size=dataset_config.batch_size,
        shuffle=dataset_config.shuffle_train,
        num_workers=dataset_config.num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=dataset_config.batch_size,
        shuffle=False,
        num_workers=dataset_config.num_workers,
    )
    return train_loader, test_loader


def build_model(model_config):
    if model_config.name == "mlp":
        return MLP(
            input_size=model_config.input_size,
            hidden_size=model_config.hidden_size,
            output_size=model_config.num_classes,
        )
    if model_config.name == "cnn":
        return CNN(num_classes=model_config.num_classes, dropout=model_config.dropout)
    raise ValueError(f"Unsupported model: {model_config.name}")


def build_optimizer(optimizer_config, parameters):
    if optimizer_config.name == "adam":
        return torch.optim.Adam(
            parameters,
            lr=optimizer_config.learning_rate,
            weight_decay=optimizer_config.weight_decay,
        )
    if optimizer_config.name == "sgd":
        return torch.optim.SGD(
            parameters,
            lr=optimizer_config.learning_rate,
            weight_decay=optimizer_config.weight_decay,
            momentum=optimizer_config.momentum,
        )
    raise ValueError(f"Unsupported optimizer: {optimizer_config.name}")
