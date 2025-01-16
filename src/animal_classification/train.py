import torch 
from torch import nn, optim
from typing import * 
from torch.utils.data import DataLoader 
import wandb 
import loguru 



model = "placeholder"
loss = nn.NLLLoss() # TODO: place holder, figure out which loss we're using.
criterion = nn.CrossEntropyLoss()



def training_step(batch)->None:
    pass

def training( batch_size:int, epochs:int, lr:float)->None: 
    
    loss = nn.NLLLoss() # TODO: place holder, figure out which loss we're using.
    criterion = nn.CrossEntropyLoss()
    train_data = ""
    dataloader = DataLoader(train_data, batch_size=batch_size)

    for idx,(images, label) in enumerate(dataloader): 
        pass

