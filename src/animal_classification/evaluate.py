import torch
import typer
from data import load_data
from model import AnimalClassifier
from omegaconf import OmegaConf
from loguru import logger


# Loading the path to the default configuration file
CONFIG_PATH = "configs/evaluate_configs.yaml"

# Setting the device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

# A pretrained model
model_checkpoint = "models/model.pth"


def evaluate(
    batch_size: int = 10, config_path: str = typer.Option(CONFIG_PATH), model_checkpoint: str = "models/model.pth"
) -> None:
    """Evaluating the trained model"""
    # Moving the model to the device
    logger.info(f"Loading model from {model_checkpoint}")
    model = AnimalClassifier().to(DEVICE)
    logger.info(f"Model loaded to {DEVICE}")
    model.load_state_dict(torch.load(model_checkpoint))
    logger.info(f"Model loaded from {model_checkpoint}")
    # Loading the parameters from the configuration file
    config = OmegaConf.load(config_path)
    batch_size = config.hyperparameters.batch_size if not batch_size else batch_size
    logger.info(f"Batch size: {batch_size}")


    # Loading the test data
    test_set = load_data(train=False)
    logger.info(f"Test set loaded from {test_set}")
    test_dataloader = torch.utils.data.DataLoader(test_set, batch_size)
    logger.info(f"Test dataloader loaded from {test_dataloader}")

    # Initiating evaluation
    model.eval()
    correct, total = 0, 0

    for img, target in test_dataloader:
        img, target = img.to(DEVICE), target.to(DEVICE)
        logger.info(f"Image and target moved to {DEVICE}")
        prediction = model(img)
        logger.info(f"Prediction: {prediction}")
        correct += (prediction.argmax(dim=1) == target).float().sum().item()
        total += target.size(0)
        logger.info(f"Correct: {correct}, Total: {total}")
    print(f"Test accuracy: {correct / total}")
    

if __name__ == "__main__":
    typer.run(evaluate)
