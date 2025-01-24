from fastapi.testclient import TestClient
import torch
from unittest.mock import patch, MagicMock
from fastapi.datastructures import UploadFile
from io import BytesIO
from src.animal_classification.api import app, preprocess_image
from src.animal_classification import api
from fastapi.testclient import TestClient


# Define the FastAPI app for testing
client = TestClient(api.app)

@patch("src.animal_classification.api.Image.open")
@patch("src.animal_classification.api.transforms.Compose")
def test_preprocess_image(mock_transforms, mock_image_open):
    # Mock PIL.Image.open
    mock_image = MagicMock()
    mock_image_open.return_value = mock_image

    # Mock resizing and mode conversion
    mock_image.resize.return_value = mock_image
    mock_image.mode = "RGB"
    mock_image.convert.return_value = mock_image

    # Mock transforms.Compose
    mock_tensor = torch.randn(1, 48, 48)  # Simulate a tensor returned by the transformation pipeline
    mock_transform_instance = MagicMock(return_value=mock_tensor)
    mock_transforms.return_value = lambda x: mock_tensor  # Return the tensor when the transform is applied

    # Create a mock UploadFile
    mock_file = BytesIO(b"fake image data")
    upload_file = UploadFile(file=mock_file, filename="test_image.jpg")

    # Call preprocess_image
    result = preprocess_image(upload_file)

    # Assert calls and processing steps
    mock_image_open.assert_called_once_with(mock_file)
    mock_image.resize.assert_called_once_with((48, 48))
    mock_image.convert.assert_called_once_with("L")
    mock_transforms.assert_called_once()

    # Check the result shape
    assert result.shape[0] == 1  # Batch dimension
    assert result.shape[1:] == (1, 48, 48)  # Grayscale image dimensions


# def test_get_prediction():
#     # Suppress Google Cloud SDK warnings
#     warnings.filterwarnings("ignore", category=UserWarning, module="google.auth._default")

#     # Mock the preprocess_image function
#     mock_preprocessed_image = torch.randn(1, 1, 48, 48)  # Simulated tensor
#     with patch("src.animal_classification.api.preprocess_image", return_value=mock_preprocessed_image):
#         # Mock the model and its behavior
#         mock_model = MagicMock()
#         mock_model.return_value = torch.tensor([[0.1, 0.2, 0.7]])  # Simulated softmax output
#         api.model = mock_model  # Assign mock model to the API module

#         # Mock the idx_to_class mapping with string keys
#         api.idx_to_class = {"0": "dog", "1": "bird", "2": "cat"}

#         # Simulate sending an image file
#         image_data = BytesIO(b"fake image data")
#         response = client.post(
#             "/get_prediction",
#             files={"image": ("test_image.jpg", image_data, "image/jpeg")},
#         )

#         # Log response for debugging
#         print("Response JSON:", response.json())

#         # Assert the response
#         assert response.status_code == 200
#         assert response.json()["prediction"] == "cat"  # Expected class

