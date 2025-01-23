from fastapi import FastAPI 
from contextlib import asynccontextmanager
from fastapi import UploadFile, File
from .model import AnimalClassifier
import torch
from .data import find_classes
from PIL import Image
import numpy as np 
import torchvision.transforms as transforms


@asynccontextmanager
async def lifespan(app: FastAPI):

    global model 
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
    if image.mode == "RGB":
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
    classes, class_to_idx = find_classes("/data")
    predicted_class_idx = int(prediction.argmax())
    predicted_class = classes[predicted_class_idx]
    return {"prediction": predicted_class}


