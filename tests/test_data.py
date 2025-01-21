import unittest
from unittest.mock import patch, MagicMock
from pathlib import Path
from src.animal_classification.data import (
    AnimalDataSet,
    normalize,
    download_data,
    find_classes,
    resize_with_padding,
    partition_dataset,
    preprocess,
    load_data,
)
from PIL import Image
import os
import tempfile
import torch


class TestAnimalDataSet(unittest.TestCase):
    def setUp(self):
        # Set up a temporary directory and sample data
        self.temp_dir = tempfile.TemporaryDirectory()
        self.dataset_path = Path(self.temp_dir.name) / "dataset"
        self.dataset_path.mkdir()

        # Create sample classes and images
        self.classes = ["dog", "cat"]
        for cls in self.classes:
            class_dir = self.dataset_path / cls
            class_dir.mkdir()
            for i in range(5):
                img = Image.new("RGB", (100, 100), color=(i * 40, i * 40, i * 40))
                img.save(class_dir / f"image_{i}.jpeg")

    def tearDown(self):
        # Clean up temporary directory
        self.temp_dir.cleanup()

    def test_animal_dataset_len(self):
        dataset = AnimalDataSet(self.dataset_path)
        self.assertEqual(len(dataset), 10)

    def test_animal_dataset_getitem(self):
        dataset = AnimalDataSet(self.dataset_path)
        img, label = dataset[0]
        self.assertIsInstance(img, Image.Image)
        self.assertIsInstance(label, int)

    def test_animal_dataset_transform(self):
        transform = lambda x: torch.tensor([1.0])
        dataset = AnimalDataSet(self.dataset_path, transform=transform)
        img, label = dataset[0]
        self.assertTrue(torch.equal(img, torch.tensor([1.0])))


class TestUtilityFunctions(unittest.TestCase):
    def test_find_classes(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            (temp_path / "dog").mkdir()
            (temp_path / "cat").mkdir()

            classes, class_to_idx = find_classes(temp_path)
            self.assertEqual(set(classes), {"dog", "cat"})
            self.assertIn("dog", class_to_idx)
            self.assertIn("cat", class_to_idx)

    def test_resize_with_padding(self):
        img = Image.new("RGB", (50, 100), color=(255, 0, 0))
        resized_img = resize_with_padding(img, 100)
        self.assertEqual(resized_img.size, (100, 100))


class TestPartitionDataset(unittest.TestCase):
    def test_partition_dataset(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            train_dir = temp_path / "train"
            test_dir = temp_path / "test"
            class_dir = temp_path / "dog"
            class_dir.mkdir()

            for i in range(10):
                img = Image.new("RGB", (100, 100), color=(255, 255, 255))
                img.save(class_dir / f"image_{i}.jpeg")

            partition_dataset(temp_dir, split_ratio=0.8)

            self.assertTrue(train_dir.exists())
            self.assertTrue(test_dir.exists())
            self.assertEqual(len(list(train_dir.glob("**/*.jpeg"))), 8)
            self.assertEqual(len(list(test_dir.glob("**/*.jpeg"))), 2)


if __name__ == "__main__":
    unittest.main()
