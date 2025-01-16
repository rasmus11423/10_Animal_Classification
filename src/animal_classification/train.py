import torch 
from torch import nn, optim
from typing import * 
from torch.utils.data import DataLoader 
import typer 

import wandb 
import loguru 

from src import load_data


model = "placeholder"
loss = nn.NLLLoss() # TODO: place holder, figure out which loss we're using.
criterion = nn.CrossEntropyLoss()



def training_step(batch)->None:
    pass

def train(batch_size:int, epochs:int, lr:float)->None: 
    

    loss = nn.NLLLoss() # TODO: place holder, figure out which loss we're using.
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(lr) #TODO: In wandb sweep, lets try with and without regularization

    train_data = load_data(train=True)
    train_dataloader = DataLoader(load_data(train=True),
                             batch_size=batch_size,
                             shuffle=True,
                            num_workers=2)

    
    for epoch in epochs:
        batch_loss = 0 

        for idx,(images, labels) in enumerate(train_dataloader):

            model.train() 

            optimizer.zero_grad()

            output = model(images) 

            loss = criterion(output, labels)
            loss.backward() 

            optimizer.step()

            batch_size+=loss.item() 


        model.eval()
        with torch.no_grad():
            # TODO: Evaluation loop goes here 

            







        


        

