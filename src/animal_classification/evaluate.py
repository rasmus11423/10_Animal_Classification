import torch
import typer
from data import load_data
from model import AnimalClassifier
#import hydra
#import os


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

model_checkpoint = "models/model.pth"

#@hydra.main(config_name = "config", config_path = f"{os.getcwd()}/configs", version_base = None)
def evaluate(model_checkpoint: str = "models/model.pth") -> None:
    """Evaluating the trained model"""

    # Moving the model to the device
    model = AnimalClassifier().to(DEVICE)
    model.load_state_dict(torch.load(model_checkpoint))

    # Loading the test data
    test_set = load_data(train=False)
    # TODO make this parameter thingy work 
    test_dataloader = torch.utils.data.DataLoader(test_set, batch_size = 10)

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


    




