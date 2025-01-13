from pathlib import Path
import typer
from torch.utils.data import Dataset
import kagglehub
from loguru import logger
import os
import json 
from PIL import Image
import mlcroissant as mlc
from kaggle.api.kaggle_api_extended import KaggleApi
from typing import Tuple, List, Dict
import torch 
import pathlib
from torchvision.transforms.functional import pad, resize
import torchvision.transforms.functional as F

class AnimalDataSet(Dataset):
    """My custom dataset."""

    def __init__(self, raw_data_path: Path, transform=None) -> None:
        self.data_path = pathlib.Path(raw_data_path) 
        self.transform = transform
        self.image_paths = list(pathlib.Path(self.data_path).glob("**/*.jpeg"))
        self.classes,self.class_to_idx = find_classes(self.data_path)

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return len(self.image_paths)

    def __getitem__(self, index: int):
        """Return a given sample from the dataset."""
        pass

    def preprocess(self, output_folder: Path) -> None:
        """Preprocess the raw data and save it to the output folder."""
        pass


    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        "Returns one sample of data, data and label (X, y)."
        img = self.load_image(index)
        class_name  = self.image_paths[index].parent.name # expects path in data_folder/class_name/image.jpeg
        class_idx = self.class_to_idx[class_name]

        # Transform
        if self.transform:
            return self.transform(img), class_idx # return data, label (X, y)
        else:
            return img, class_idx # return data, label (X, y)



    def load_image(self, index): 
        image_path = self.image_paths[index]
        return Image.open(image_path) 

def download_data(raw_data_path: str="./data/raw"):
    raw_data_path = os.path.abspath(raw_data_path)
    os.makedirs(raw_data_path, exist_ok=True)


    api = KaggleApi()

    try: 
        api.authenticate()
    except Exception as e: 
        raise f"Encountered error: {e}, please make sure to set your kaggle secret token."

    if len(os.listdir(raw_data_path)) > 1:
        logger.info("Data already exists. Skipping download.")
        return

    logger.info("Downloading the dataset from Kaggle...")
    api.dataset_download_files(
        "alessiocorrado99/animals10", path=raw_data_path, unzip=True
    )
    logger.info(f"Dataset successfully downloaded to: {raw_data_path}")



def find_classes(directory: str) -> Tuple[List[str], Dict[str, int]]:

    classes = os.listdir(directory) # get all the class names 
    class_to_idx = {class_name: idx for idx,class_name in enumerate(classes)}
    return classes, class_to_idx 


from PIL import Image
import torchvision.transforms.functional as F

def resize_with_padding(image: Image.Image, target_size: int, padding_color=(255, 255, 255)) -> Image.Image:
    original_width, original_height = image.size

    scale = min(target_size / original_width, target_size / original_height)

    new_width = int(original_width * scale)
    new_height = int(original_height * scale)

    resized_image = F.resize(image, (new_height, new_width))

    delta_width = target_size - new_width
    delta_height = target_size - new_height

    padding = (
        delta_width // 2,  # Left
        delta_height // 2,  # Top
        delta_width - (delta_width // 2),  # Right
        delta_height - (delta_height // 2)  # Bottom
    )

    squared_image = F.pad(resized_image, padding, fill=padding_color)

    return squared_image



def preprocess() -> None:
    output_folder = "data/raw"
    download_data(output_folder)
    logger.info("Preprocessing data...")

    logger.info("initiating dataset class")
    dataset = AnimalDataSet("data/raw/raw-img")
    logger.info("Indexing a class") 

    img, label = dataset[0] 
    print(f"label: {img}")

    logger.info("Iterating over the data") 
    
    print(len(dataset))

    logger.info("Resizing the data") 

    
    for idx, (img, label_idx) in enumerate(dataset): 
        img_resized = resize_with_padding(img, 48) 
        label = dataset.classes[label_idx]
        class_folder = f"data/processed/{label}"
        os.makedirs(class_folder, exist_ok=True)
        save_path = os.path.join(class_folder, f"{idx}.jpeg")

        img_resized.save(save_path, format="JPEG") 
    logger.info("All images saved succuesfully")






if __name__ == "__main__":
    typer.run(preprocess)