import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Normalize
from pathlib import Path
from typing import Tuple
from PIL import Image
import os

from model import Animal_classifier
from data import AnimalDataSet

def train_model(model, train_loader, criterion, optimizer, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()

        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}")


if __name__ == '__main__':
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Dataset paths
    data_path = "data/processed"
    transform = Compose([
        ToTensor(),
        Normalize((0.5,), (0.5,))
    ])

    # Dataset and DataLoader
    dataset = AnimalDataSet(data_path, transform=transform)
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Model, Loss, and Optimizer
    model = Animal_classifier().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    train_model(model, train_loader, criterion, optimizer, num_epochs=10)

    # Save the model
    torch.save(model.state_dict(), "animal_classifier.pth")
    print("Model saved successfully.")