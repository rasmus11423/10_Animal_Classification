import torch
import typer
from data import load_data
from model import AnimalClassifier
from omegaconf import OmegaConf

# Loading the path to the default configuration file
CONFIG_PATH = "configs/evaluate_configs.yaml"

# Setting the device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

# A pretrained model
model_checkpoint = "models/model.pth"

def evaluate(batch_size: int = 10, config_path: str = typer.Option(CONFIG_PATH), model_checkpoint: str = "models/model.pth") -> None:
    """Evaluating the trained model"""
    # Moving the model to the device
    model = AnimalClassifier().to(DEVICE)
    model.load_state_dict(torch.load(model_checkpoint))
    # Loading the parameters from the configuration file
    config = OmegaConf.load(config_path)
    batch_size = config.hyperparameters.batch_size if not batch_size else batch_size

    print(batch_size)

    # Loading the test data
    test_set = load_data(train=False)
    test_dataloader = torch.utils.data.DataLoader(test_set, batch_size)

    # Initiating evaluation
    model.eval()
    correct, total = 0,0

    for img, target in test_dataloader:
        img, target = img.to(DEVICE), target.to(DEVICE)
        prediction = model(img)
        correct += (prediction.argmax(dim=1) == target).float().sum().item()
        total += target.size(0)
    print(f"Test accuracy: {correct / total}")

if __name__ == "__main__":
    typer.run(evaluate)


    




