import unittest
import torch
import torch.nn as nn
from src.animal_classification.model import AnimalClassifier

class TestAnimalClassifier(unittest.TestCase):
    def setUp(self):
        self.model = AnimalClassifier()

    def test_model_output_shape(self):
        """Test if the model produces the correct output shape."""
        input_tensor = torch.randn(1, 1, 48, 48)  # Batch size 1, 1 channel, 48x48 image
        output = self.model(input_tensor)
        self.assertEqual(output.shape, (1, 10))  # Expecting 10 output classes

    def test_model_parameters(self):
        """Test if the model parameters are trainable."""
        for param in self.model.parameters():
            self.assertTrue(param.requires_grad)

    def test_forward_pass(self):
        """Test a single forward pass through the model."""
        input_tensor = torch.randn(5, 1, 48, 48)  # Batch size 5
        output = self.model(input_tensor)
        self.assertEqual(output.shape, (5, 10))  # Batch size 5, 10 output classes

    def test_no_nan_outputs(self):
        """Ensure that the model does not produce NaN values in the output."""
        input_tensor = torch.randn(1, 1, 48, 48)
        output = self.model(input_tensor)
        self.assertFalse(torch.isnan(output).any())

    def test_model_trainable_layers(self):
        """Ensure all layers of the model are trainable."""
        for name, param in self.model.named_parameters():
            with self.subTest(layer=name):
                self.assertTrue(param.requires_grad)

if __name__ == "__main__":
    unittest.main()
