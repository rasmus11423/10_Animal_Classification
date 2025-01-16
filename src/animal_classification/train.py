import torch
from torch import nn, optim
from typing import Tuple
from torch.utils.data import DataLoader
import typer

import wandb
from loguru import logger

from src import load_data


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")


model = "placeholder.device()"


def training_step(
    images: torch.Tensor, labels, model: nn.Module, criterion: nn.Module, device: torch.device, optimizer: torch.optim
) -> Tuple[float, float, float]:
    optimizer.zero_grad()
    output = model(images)
    loss = criterion(output, labels)
    loss.backward()
    optimizer.step()

    _, predicted = torch.max(output.data, 1)
    correct = (predicted == labels).sum().item()
    accuracy = 100 * correct / labels.size(0)

    return loss.item(), accuracy, predicted


def evaluate(model: nn.Module, dataloader: DataLoader, criterion: nn.Module) -> Tuple[float, float]:
    total_loss = 0
    correct = 0
    total = 0

    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    avg_loss = total_loss / len(dataloader)
    accuracy = 100 * correct / total

    return avg_loss, accuracy


def train(batch_size: int, epochs: int, lr: float) -> None:
    logger.info("Initializing wandb project...")
    wandb.init(
        project="MLops-animal-project",
        entity="grp-44",
        config={"learning_rate": lr, "epochs": epochs, "batch_size": batch_size},
    )

    mytable = wandb.Table(columns=["images", "label", "Predictions", "epoch"])

    wandb.watch(model, log_freq=300)  # <- thought it would be cool to track the gradients

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr)  # TODO: In wandb sweep, lets try with and without regularization

    train_data = load_data(train=True)
    test_data = load_data(train=False)

    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers=2)

    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=2)

    statistics = {"train_loss": [], "train_accuracy": [], "validation_loss": [], "validation_accuracy": []}

    for epoch in epochs:
        run_loss = 0
        run_acc = 0

        for idx, (images, labels) in enumerate(train_dataloader):
            images, labels = images.to(device), labels.to(device)

            model.train()
            batch_loss, batch_acc, predictions = training_step(images, labels, model, criterion, optimizer)

            run_acc += batch_acc
            run_loss += batch_loss

        avg_loss, avg_acc = run_loss / len(train_dataloader), run_acc / len(train_dataloader)
        statistics["train_loss"].append(avg_loss)
        statistics["train_accuracy"].append(avg_acc)

        model.eval()
        with torch.no_grad():
            val_loss, val_accuracy = evaluate(model, test_dataloader, criterion)
            statistics["validation_loss"].append(val_loss)
            statistics["validation_accuracy"].append(val_accuracy)

            # We are going to log random images, the randomness is determined by the random index between 0 and len(datasetloader)
            random_indices = torch.randint(0, len(images), (5,))  # Get 5 random indices

            for idx in random_indices:
                mytable.add_data(
                    wandb.Image(images[idx].numpy().cpu()), labels[idx].item(), predictions[idx].item(), epoch
                )

        wandb.log(
            {
                "Train accuracy": avg_acc,
                "Train loss": avg_loss,
                "validation loss": val_loss,
                "validation accuracy": val_accuracy,
                "epoch": epoch,
                "predictions": mytable,
            }
        )

        print("----------------------" * 5)
        print(f"Epoch: {epoch}")
        print(f"Train loss: {avg_loss}")
        print(f"Train accuracy: {avg_acc}")
        print(f"val loss: {val_loss}")
        print(f"val accuracy: {val_accuracy}")


# ----------------------------
# TODO: Generate plots and figures


if __name__ == "__main__":
    typer.run(train)
