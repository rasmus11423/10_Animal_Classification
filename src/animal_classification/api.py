from fastapi import FastAPI 
from contextlib import asynccontextmanager
from fastapi import UploadFile, File, BackgroundTasks, HTTPException
from .model import AnimalClassifier
import torch
from PIL import Image
import numpy as np 
import torchvision.transforms as transforms
from google.cloud import storage
import datetime
import json
import uuid
from io import BytesIO
from prometheus_client import Counter, make_asgi_app
from loguru import logger 

BUCKET_NAME = "dtumlops_databucket"

# Define Prometheus metrics
error_counter = Counter("prediction_error", "Number of prediction errors")


@asynccontextmanager
async def lifespan(app: FastAPI):

    global model, idx_to_class

    idx_to_class= {
        "0": "butterfly",
        "1": "cat",
        "2": "chicken",
        "3": "cow",
        "4": "dog",
        "5": "elephant",
        "6": "horse",
        "7": "sheep",
        "8": "spider",
        "9": "squirrel"
    }

    MODEL_PATH = "models/model.pth"
    model = AnimalClassifier()
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()
    yield
    print("Shutting down...")
    del model 


def preprocess_image(image: UploadFile):
    image = Image.open(image.file)
    image = image.resize((48, 48))
    if image.mode in ["RGB", "RGBA"]:
        image = image.convert("L")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: (x - x.mean()) / x.std())
    ])

    return transform(image).unsqueeze(0) # batch_dimension

app  = FastAPI(lifespan=lifespan) 
app.mount("/metrics", make_asgi_app())

def save_image_to_gcp(image: UploadFile, image_id: str):
    """Save the uploaded image to GCP bucket."""
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    
    image.file.seek(0)
    # Read the image content **before** using it in a background task
    image_content = image.file.read()  # Read the file before it gets closed

    # Create an in-memory file object
    image_buffer = BytesIO(image_content)

    blob = bucket.blob(f"User_data/images/{image_id}.jpg")
    image_buffer.seek(0) 
    blob.upload_from_file(image_buffer, content_type=image.content_type)
    print(f"Image {image_id}.jpg uploaded to GCP bucket.")


def save_prediction_to_gcp(image_id: str, label: str):
    """Save the prediction results to GCP bucket."""
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)

    time = datetime.datetime.now(tz=datetime.timezone.utc)
    data = {
        "image_name": image_id,
        "label": label,
        "timestamp": time.isoformat(),
    }

    blob = bucket.blob(f"User_data/predictions/{image_id}.jpg")
    blob.upload_from_string(json.dumps(data))
    print(f"Prediction of {image_id}.jpg saved to GCP bucket.")


@app.post("/get_prediction")
async def get_prediction(image: UploadFile = File(...)):
    try:
        image_processed = preprocess_image(image) 
        prediction = model(image_processed)
        predicted_class_idx = int(prediction.argmax())
        predicted_class = idx_to_class[str(predicted_class_idx)]
        image_id = str(uuid.uuid4())
        save_image_to_gcp(image, image_id)
        save_prediction_to_gcp(image_id, predicted_class)
        return {"prediction": predicted_class}
    
    except Exception as e:
        error_counter.inc()
        raise HTTPException(status_code=500, detail=str(e)) from e

    

