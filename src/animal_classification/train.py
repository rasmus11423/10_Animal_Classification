import torch
from torch import nn, optim
from typing import Tuple
from torch.utils.data import DataLoader
from accelerate import Accelerator

import typer
from omegaconf import OmegaConf

import wandb
from loguru import logger

from torch.profiler import profile, ProfilerActivity, tensorboard_trace_handler
from data import load_data, download_processed_data
from model import AnimalClassifier


# Path to default configuration file
CONFIG_PATH = "configs/training_configs.yaml"





def training_step(
    images: torch.Tensor, labels, model: nn.Module, criterion: nn.Module, optimizer: torch.optim.Optimizer, accelerator: Accelerator, device: Accelerator.device
) -> Tuple[float, float, float]:
    optimizer.zero_grad()
    output = model(images)
    loss = criterion(output, labels)
    accelerator.backward(loss)
    optimizer.step()

    _, predicted = torch.max(output.data, 1)
    correct = (predicted == labels).sum().item()
    accuracy = 100 * correct / labels.size(0)

    return loss.item(), accuracy, predicted


def evaluate(model: nn.Module, dataloader: DataLoader, criterion: nn.Module, device: Accelerator.device) -> Tuple[float, float]:
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


def train(
    batch_size: int = 130,
    epochs: int = 25,
    lr: float = 1e-2,
    optimizer_name: str = None,
    criterion_name: str = None,
    config_path: str = typer.Option(CONFIG_PATH),
) -> None:
    """Training the model on the animal data set."""
    
    profiler = profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]
        if torch.cuda.is_available()
        else [ProfilerActivity.CPU],
        on_trace_ready=tensorboard_trace_handler("runs/profiler_logs"),
        with_stack=True,
        record_shapes=True,
        profile_memory=True,
        schedule=torch.profiler.schedule(
            wait=2,  # Wait for 2 iterations before profiling
            warmup=1,  # Warm up for 2 iterations
            active=5,  # Profile for 5 iterations
            repeat=1,  # Repeat profiling once
        ),
    )

    # Download preprocessed data at the start
    download_processed_data(
        bucket_name="dtumlops_databucket",
        source_path="data",
        local_path=""
    )


    accelerator = Accelerator()
    # accelerator handles the device configuration
    device = accelerator.device
    logger.info(f"Using device: {device}")

    model = AnimalClassifier()
    model = model.to(device)

    
    # Loading parameters from configuration file
    config = OmegaConf.load(config_path)
    logger.info(f"Config loaded from {config_path}")
    batch_size = config.hyperparameters.batch_size if not batch_size else batch_size
    epochs = config.hyperparameters.epochs if not epochs else epochs
    lr = config.optimizer.lr if not lr else lr
    optimizer_name = config.optimizer.name if not optimizer_name else optimizer_name
    criterion_name = config.criterion if not criterion_name else criterion_name

    logger.info("Initializing wandb project...")
    # Initializing the wandb project
    wandb.init(
        project="MLops-animal-project",
        config={
            "learning_rate": lr,
            "epochs": epochs,
            "batch_size": batch_size,
            "optimizer_name": optimizer_name,
            "criterion_name": criterion_name,
        },
    )

    table = wandb.Table(columns=["image", "prediction", "label"])


    wandb.watch(model, log_freq=300)  # Track gradients in W&B

    # Initializing optimizer and criterion
    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)
    criterion = getattr(nn, criterion_name)()

    train_data = load_data(rgb=False, train=True)
    test_data = load_data(rgb=False, train=False)

    test_dataloader = DataLoader(
        test_data, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=0,  # Parallel loading
        pin_memory=True  # Faster data transfer to GPU
    )
    train_dataloader = DataLoader(
        train_data, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )

    model, optimizer, criterion, train_dataloader, test_dataloader = accelerator.prepare(model,
                                                                                          optimizer, 
                                                                                          criterion, 
                                                                                          train_dataloader, 
                                                                                          test_dataloader
                                                                                          )
        
    
    logger.info("Starting training")
    profiler.start()

    best_val_acc = 0
    
    for epoch in range(epochs):
        run_loss = 0
        run_acc = 0
        for idx, (images, labels) in enumerate(train_dataloader):
           # images, labels = images.to(device), labels.to(device)

            model.train()

            batch_loss, batch_acc, predictions = training_step(images, labels, model, criterion, optimizer, accelerator, device)

            run_acc += batch_acc
            run_loss += batch_loss

            profiler.step()  # Step the profiler after each iteration

        # table to add images, predictions and labels. only for every second epoch and only 5 images
        if epoch % 2 == 0:
            for i in range(5):
                # Convert tensors to numpy arrays or primitive types
                img_np = images[i].cpu().numpy()
                pred_val = predictions[i].item()  # Convert to Python scalar
                label_val = labels[i].item()  # Convert to Python scalar
                table.add_data(img_np, pred_val, label_val)

        avg_loss, avg_acc = run_loss / len(train_dataloader), run_acc / len(train_dataloader)
        logger.info(f"Epoch {epoch + 1}, Loss: {avg_loss:.4f}, Accuracy: {avg_acc:.2f}%")

        model.eval()

        with torch.no_grad():
            val_loss, val_accuracy = evaluate(model, test_dataloader, criterion, device)
            logger.info(f"Validation loss: {val_loss:.4f}, Validation accuracy: {val_accuracy:.2f}%")

            if val_accuracy > best_val_acc:
                best_val_acc = val_accuracy
                logger.info("New best validation accuracy achieved")



        wandb.log(
            {
                "Train accuracy": avg_acc,
                "Train loss": avg_loss,
                "Validation loss": val_loss,
                "Validation accuracy": val_accuracy,
                "epoch": epoch,
                
            }
        )

        if epoch % 2 == 0:
            wandb.log({"train_images": table})



    profiler.stop()

    artifacts=wandb.Artifact("model", type="model", description="Model for animal classification", metadata={"epochs": epochs, "batch_size": batch_size, "optimizer_name": optimizer_name, "criterion_name": criterion_name, "accuracy": val_accuracy})
    artifacts.add_file("models/model.pth")
    logger.info("Logging artifact to wandb")
    wandb.log_artifact(artifacts)

    
    logger.info("Training completed. Check TensorBoard for profiler logs.")
    logger.info("Run: tensorboard --logdir runs/profiler_logs")
    logger.info(f"unwrapped model: {model}")
    unwrapped_model = accelerator.unwrap_model(model)
    torch.save(unwrapped_model.state_dict(), "models/model.pth")
    logger.info("Model saved to models/model.pth")

if __name__ == "__main__":
    typer.run(train)
