from pathlib import Path
import typer
from torch.utils.data import Dataset
from loguru import logger
import os
from typing import Tuple, List, Dict
import pathlib
import torchvision.transforms.functional as F
import torch
from torchvision import transforms
from PIL import Image
import math
from google.cloud import storage


translate = {
    "cane": "dog",
    "elefante": "elephant",
    "farfalla": "butterfly",
    "gallina": "chicken",
    "gatto": "cat",
    "mucca": "cow",
    "pecora": "sheep",
    "scoiattolo": "squirrel",
    "dog": "cane",
    "cavallo": "horse",
    "elephant": "elefante",
    "butterfly": "farfalla",
    "chicken": "gallina",
    "cat": "gatto",
    "cow": "mucca",
    "ragno": "spider",
    "squirrel": "scoiattolo",
}


def normalize(images: torch.Tensor) -> torch.Tensor:
    return (images - images.mean()) / images.std()


class AnimalDataSet(Dataset):
    """My custom dataset."""

    def __init__(self, raw_data_path: Path, transform=None) -> None:
        self.data_path = pathlib.Path(raw_data_path)
        self.transform = transform
        self.image_paths = list(pathlib.Path(self.data_path).glob("**/*.jpeg"))
        self.classes, self.class_to_idx = find_classes(self.data_path)

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return len(self.image_paths)

    def preprocess(self, output_folder: Path) -> None:
        """Preprocess the raw data and save it to the output folder."""
        pass

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        "Returns one sample of data, data and label (X, y)."
        img = self.load_image(index)
        class_name = self.image_paths[index].parent.name  # expects path in data_folder/class_name/image.jpeg
        class_idx = self.class_to_idx[class_name]

        # Transform
        if self.transform:
            return self.transform(img), class_idx  # return data, label (X, y)
        else:
            return img, class_idx  # return data, label (X, y)

    def load_image(self, index: int) -> Image:
        image_path = self.image_paths[index]
        return Image.open(image_path)

def download_data(raw_data_path: str = "./data/raw") -> None:
    raw_data_path = os.path.abspath(raw_data_path)
    os.makedirs(raw_data_path, exist_ok=True)

    if len(os.listdir(raw_data_path)) > 1:
        logger.info("Data already exists. Skipping download.")
        return

    # Only import and authenticate when needed
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()

    try:
        api.authenticate()
    except Exception as e:
        raise Exception(f"Encountered error: {e}, please make sure to set your Kaggle secret token.")

    logger.info("Downloading the dataset from Kaggle...")
    api.dataset_download_files("alessiocorrado99/animals10", path=raw_data_path, unzip=True)
    logger.info(f"Dataset successfully downloaded to: {raw_data_path}")



def find_classes(directory: str) -> Tuple[List[str], Dict[str, int]]:
    classes = os.listdir(directory)  # get all the class names
    class_to_idx = {class_name: idx for idx, class_name in enumerate(classes)}
    return classes, class_to_idx


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
        delta_height - (delta_height // 2),  # Bottom
    )

    squared_image = F.pad(resized_image, padding, fill=padding_color)
    return squared_image


def partition_dataset(folder: str = "data/processed/", split_ratio: float = 0.8) -> None:
    """
    Split the dataset into train and test splits
    """
    # First, create train and test directories at the root level
    train_dir = os.path.join(folder, "train")
    test_dir = os.path.join(folder, "test")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    classes_folder = [
        f for f in os.listdir(folder) if os.path.isdir(os.path.join(folder, f)) and f not in ["train", "test"]
    ]

    for label in classes_folder:
        os.makedirs(os.path.join(train_dir, label), exist_ok=True)
        os.makedirs(os.path.join(test_dir, label), exist_ok=True)

        source_folder = os.path.join(folder, label)
        images_paths = list(pathlib.Path(source_folder).glob("*.jpeg"))
        total_images = len(images_paths)
        num_train = math.floor(total_images * split_ratio)

        for idx, image_path in enumerate(images_paths):
            image = Image.open(image_path)
            filename = image_path.name

            if idx < num_train:
                save_path = os.path.join(train_dir, label, filename)
            else:
                save_path = os.path.join(test_dir, label, filename)

            image.save(save_path)
            image.close()

        import shutil

        shutil.rmtree(source_folder)


def download_processed_data(bucket_name: str, source_path: str, local_path: str = "data/processed") -> None:
    """Downloads preprocessed data from GCS to local storage."""
    logger.info(f"Downloading preprocessed data from GCS bucket {bucket_name}")
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    
    # Create local directory
    os.makedirs(local_path, exist_ok=True)
    
    # Download all files
    blobs = bucket.list_blobs(prefix=source_path)
    for blob in blobs:
        if blob.name.endswith('/'):  # Skip directories
            continue
        # Get relative path
        rel_path = blob.name[len(source_path):].lstrip('/')
        # Construct local file path
        local_file_path = os.path.join(local_path, rel_path)
        # Create directories if they don't exist
        os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
        # Download file
        blob.download_to_filename(local_file_path)
    
    logger.info(f"Data downloaded to {local_path}")


def load_data(rgb=False, train=True):
    if rgb:
        transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: (x - x.mean()) / x.std())])

    else:
        transform = transforms.Compose(
            [transforms.Grayscale(1), transforms.ToTensor(), transforms.Lambda(lambda x: (x - x.mean()) / x.std())]
        )

    if train:
        return AnimalDataSet("data/processed/train", transform)

    return AnimalDataSet("data/processed/test", transform)


def preprocess() -> None:
    output_folder = "data/raw"
    download_data(output_folder)
    logger.info("Preprocessing data...")

    logger.info("initiating dataset class")
    dataset = AnimalDataSet("data/raw/raw-img")
    logger.info("Indexing a class")

    img, label = dataset[0]
    logger.info("Resizing the data")

    for idx, (img, label_idx) in enumerate(dataset):
        img_resized = resize_with_padding(img, 48)
        label = dataset.classes[label_idx]
        label_english = translate[label]
        class_folder = f"data/processed/{label_english}"
        os.makedirs(class_folder, exist_ok=True)
        save_path = os.path.join(class_folder, f"{idx}.jpeg")
        img_resized.save(save_path)

        # Now we save

    logger.info("All images saved succuesfully")
    logger.info("Partitioning test and train data set...")
    partition_dataset()
    logger.info("Images partitioned...")


if __name__ == "__main__":
    typer.run(preprocess)
