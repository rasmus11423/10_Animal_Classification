from fastapi import FastAPI 
from contextlib import asynccontextmanager
from fastapi import UploadFile, File
from .model import AnimalClassifier
import torch
from PIL import Image
import numpy as np 
import torchvision.transforms as transforms


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


@app.post("/get_prediction")
async def get_prediction(image: UploadFile = File(...)):


    image = preprocess_image(image) 
    prediction = model(image)
    predicted_class_idx = int(prediction.argmax())
    predicted_class = idx_to_class[predicted_class_idx]
    return {"prediction": predicted_class}


@app.get("/")
async def read_root():
    return {"message": "Welcome to the MNIST model inference API!"}

