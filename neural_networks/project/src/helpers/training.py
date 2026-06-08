from collections.abc import Callable
from pathlib import Path

import pandas as pd
import torch


def train_epoch(model, loader, criterion, optimizer, device, flatten=False):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        if flatten:
            images = images.view(images.size(0), -1)
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += images.size(0)

    return total_loss / total, correct / total


def evaluate(model, loader, criterion, device, flatten=False):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for images, labels in loader:
            if flatten:
                images = images.view(images.size(0), -1)
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += images.size(0)

    return total_loss / total, correct / total


def train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    device,
    epochs,
    flatten=False,
    log_every=1,
    logger: Callable[[str], None] = print,
):
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, flatten=flatten
        )
        val_loss, val_acc = evaluate(
            model, val_loader, criterion, device, flatten=flatten
        )

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        if logger and (epoch == 1 or epoch % log_every == 0):
            logger(
                f"Epoch {epoch:3d}/{epochs} | "
                f"Train loss: {train_loss:.4f}  acc: {train_acc:.4f} | "
                f"Val loss: {val_loss:.4f}  acc: {val_acc:.4f}"
            )

    return history


def save_model_checkpoint(
    model,
    experiment,
    output_dir,
    history=None,
    extra_metadata=None,
):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    checkpoint_path = output_path / f"{experiment.name}.pt"
    metadata = {
        "experiment_name": experiment.name,
        "model": experiment.model.__dict__,
        "optimizer": experiment.optimizer.__dict__,
        "dataset": experiment.dataset.__dict__,
        "epochs": experiment.epochs,
        "seed": experiment.seed,
        "device": experiment.device,
    }
    if history is not None:
        metadata["history"] = history
    if extra_metadata:
        metadata.update(extra_metadata)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "metadata": metadata,
        },
        checkpoint_path,
    )
    return checkpoint_path


def history_to_rows(experiment, history):
    rows = []
    for index, (train_loss, val_loss, train_acc, val_acc) in enumerate(
        zip(
            history["train_loss"],
            history["val_loss"],
            history["train_acc"],
            history["val_acc"],
        ),
        start=1,
    ):
        rows.append(
            {
                "name": experiment.name,
                "model": experiment.model.name,
                "epoch": index,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "train_acc": train_acc,
                "val_acc": val_acc,
                "optimizer": experiment.optimizer.name,
                "learning_rate": experiment.optimizer.learning_rate,
                "momentum": experiment.optimizer.momentum,
                "weight_decay": experiment.optimizer.weight_decay,
            }
        )
    return rows


def save_training_tables(results, history_rows, results_csv, history_csv):
    results_path = Path(results_csv)
    history_path = Path(history_csv)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    history_path.parent.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(results).to_csv(results_path, index=False)
    pd.DataFrame(history_rows).to_csv(history_path, index=False)
    return results_path, history_path
