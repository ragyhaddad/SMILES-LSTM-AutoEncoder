#!/bin/bash/python 
# Author: Ragy Haddad <ragy@deepbiometherapeutics.com>
import sys,os 
import torch
from torch import nn 
import torch.nn.functional as F 
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence,pad_sequence 
from torch.utils.tensorboard import SummaryWriter # Tensorboard 
from config import Config 
from dataloader import Dataset 
from tokenizer import Tokenizer 
from tqdm import tqdm 
from model import AAE 


# Train each epoch and calculate loss on UNPADDED SEQEUNCES 
def train_epoch(epoch,model,optimizer,criterion,dataloader,batch_size,device,writer): 
    loss = 0
    for idx,data in enumerate(dataloader.load_batches(batch_size=batch_size,label_encoding=True,shifted_targets=True,padding=True,return_length=True)):
        loss = 0 # Per Batch Loss 
        optimizer.zero_grad()
        x,y,l_x,l_y = data   # src,target,src_length,trg_length  
        x,y = torch.from_numpy(x).long().to(device),torch.from_numpy(y).long().to(device)
        l_x,l_y = torch.from_numpy(l_x).to(device),torch.from_numpy(l_y).to(device)
        encoder_inputs = (x,l_x) 
        latent_codes = model.encoder_forward(x,l_x)

        # Initial Decoder to All Zeros 
        decoder_inputs_init = torch.zeros_like(y).to(device)
        decoder_inputs = (decoder_inputs_init,l_y)
        decoder_outputs, decoder_output_lengths, _ = model.decoder_forward(
                    *decoder_inputs, latent_codes, is_latent_states=True)
        decoder_targets = (y,l_y) # True Shifted Outputs  
        # Unpad outputs for loss calculation -- Important!!
        unpadded_ = []
        for t,l in zip(decoder_outputs,decoder_output_lengths):
            unpadded_.append(t[:l]) 
        decoder_outputs = torch.cat(unpadded_,dim=0)
        # Unpad Targets 
        decoder_targets = torch.cat(
                    [t[:l] for t, l in zip(*decoder_targets)], dim=0)  
        loss += criterion(decoder_outputs,decoder_targets)
        loss.backward(retain_graph=True)
        optimizer.step()
        print(epoch,loss.item())
    writer.add_scalar('Loss/Train',loss,epoch)
    return loss

def train(model,optimizer,dataloader,config,device):
    writer = SummaryWriter()
    epochs = 200
    criterion = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        loss = train_epoch(epoch,model,optimizer,criterion,dataloader,config.batch_size,device,writer)
        torch.save(model.state_dict(),'models/aae_new_epoch_%i.pth' % epoch)

def main():
    # Configuration
    config = Config()
    # Device
    device = torch.device(config.device)
    # Model  
    model = AAE(config).train().to(device) 
    # Optimizer
    optimizer = torch.optim.Adam(list(model.encoder.parameters()) +
                                     list(model.decoder.parameters()),
                                     lr=config.lr) 
    # Data 
    dataloader = Dataset(sys.argv[1],sep='\t') 

    #### TRAIN  
    train(model,optimizer,dataloader,config,device)  

if __name__ == "__main__":
    main()