from google.cloud import storage
from PIL import Image
import io
from torchvision import transforms
import torch
from transformers import CLIPModel, CLIPProcessor
from evidently.metrics import DataDriftTable
from evidently.report import Report
import numpy as np
import pandas as pd
import webbrowser
import os

BUCKET_NAME = "dtumlops_databucket"


# Transforming function to correct formal for CLIP
transform = transforms.Compose([
    transforms.Resize((48, 48)),
    transforms.Lambda(lambda x: x.convert("RGB")),
    transforms.ToTensor()
])

# Load CLIP model and processor
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def load_images_from_gcp(folder):
    "Load images from google cloud bucket."
    # Initiating the bucket
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    
    # Downloading the images from the folders
    images = []
    blob_current = list(bucket.list_blobs(prefix=folder))
    for blob in blob_current:
        if not blob.name.lower().endswith((".jpg", ".jpeg", ".png")):
            print(f"Skipping non-image file: {blob.name}")
            continue
        img_bytes = blob.download_as_bytes()
        img = Image.open(io.BytesIO(img_bytes))

        # Transforming images
        img = transform(img)
        images.append(img)
    return images

def extract_clip_features(images):
    """Extract CLIP image features from a list of images."""
    batch_size = 32  # Process images in batches
    features = []
    for i in range(0, len(images), batch_size):
        batch = images[i:i + batch_size]
        inputs = processor(images=batch, return_tensors="pt", padding=True)
        with torch.no_grad():
            img_features = model.get_image_features(inputs["pixel_values"])
        features.append(img_features.cpu().numpy())  # Convert to NumPy
    return np.vstack(features)


# Load images from GCP

# Defining the bucket folders
current_folder = "User_data/images"
reference_folder = "data/processed/test/butterfly"

# Extracting the current and reference
images_current = load_images_from_gcp(current_folder)
images_reference = load_images_from_gcp(reference_folder)

# Extracting features
current_features = extract_clip_features(images_current)
reference_features = extract_clip_features(images_reference)

# Converting to dataframe
feature_columns = [f"CLIP_Feature_{i}" for i in range(reference_features.shape[1])]
reference_df = pd.DataFrame(reference_features, columns=feature_columns)
current_df = pd.DataFrame(current_features, columns=feature_columns)

report = Report(metrics=[DataDriftTable()])
report.run(reference_data=reference_df, current_data=current_df)
report.save_html("clip_data_drift.html")

file_path = os.path.abspath("clip_data_drift.html")
webbrowser.open("file://" + file_path)
