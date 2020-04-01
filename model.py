#!/bin/bash/python 
# Author: Ragy Haddad <ragy@deepbiometherapeutics.com>
import sys,os 
import torch
from torch import nn 
import torch.nn.functional as F 
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence,pad_sequence
from config import Config 
from dataloader import Dataset 
from tokenizer import Tokenizer 

# Cited + Mods -- Standard AAE Implementation + Sampling and packed sequences for fast batch training 
# https://github.com/molecularsets/moses/blob/master/moses/aae/model.py
 

# Note: Embedding Layer is Shared Between Encoder and Decoder
class Encoder(nn.Module):
    def __init__(self,embedding_layer,config):
        super(Encoder,self).__init__()
        self.embedding_layer = config.embedding_size 
        self.embedding = embedding_layer
        self.lstm_layer = nn.LSTM(config.embedding_size,config.hidden_size,num_layers=config.num_layers,batch_first=True,bidirectional=config.bidirectional) 
        self.linear_layer = nn.Linear((int(config.bidirectional) + 1) * config.num_layers * config.hidden_size,config.latent_size)  
    def forward(self,x,lengths):
        batch_size = x.shape[0] 
        x = self.embedding(x)
        x = pack_padded_sequence(x, lengths, batch_first=True,enforce_sorted=False)
        _, (_, x) = self.lstm_layer(x) # Get Cell State 
        x = x.permute(1, 2, 0).contiguous().view(batch_size, -1) # Reshape for Linear
        x = self.linear_layer(x) 
        return x # Latent Vector -- Cell State
 
class Decoder(nn.Module):
    def __init__(self,embedding_layer,config):
        super(Decoder,self).__init__()
        self.config = config
        self.latent_to_hidden = nn.Linear(config.latent_size,config.hidden_size)
        self.embedding = embedding_layer 
        self.lstm = nn.LSTM(config.embedding_size,config.hidden_size,num_layers=config.num_layers,batch_first=True)
        self.linear_layer = nn.Linear(config.hidden_size,config.vocab_size) # Logits with size of vocab 
    def forward(self,x,lengths,states,is_latent_states=False):
        if is_latent_states:
            c0 = self.latent_to_hidden(states) 
            # Stack Cells for each layer
            c0 = c0.unsqueeze(0).repeat(self.config.num_layers, 1, 1) 
            h0 = torch.zeros_like(c0) # Hidden with All Zeros - same size as cell 
            states = (h0, c0) 
        x = self.embedding(x) 
        x = pack_padded_sequence(x, lengths, batch_first=True,enforce_sorted=False)   
        x, states = self.lstm(x, states)
        x, lengths = pad_packed_sequence(x, batch_first=True)
        x = self.linear_layer(x) 
        return x,lengths,states 

class Discriminator(nn.Module):
    def __init__(self,config):
        super(Discriminator, self).__init__()
        input_size = config.latent_size 
        layers = config.discriminator_layers 
        in_features = [input_size] + layers
        out_features = layers + [1] # for BCELoss 
        self.layers_seq = nn.Sequential()
        for k, (i, o) in enumerate(zip(in_features, out_features)):
            self.layers_seq.add_module('linear_{}'.format(k), nn.Linear(i, o))
            if k != len(layers):
                self.layers_seq.add_module('activation_{}'.format(k),
                                           nn.ELU(inplace=True))
    def forward(self, x):
        return self.layers_seq(x)  

# Main Class
class AAE(nn.Module):
    def __init__(self,config):
        super(AAE,self).__init__() 
        self.config = config 
        self.latent_size = config.latent_size 
        self.device = config.device 
        self.embedding_layer = nn.Embedding(config.vocab_size,config.embedding_size,padding_idx=config.padding_int_value)
        self.encoder = Encoder(self.embedding_layer,config)
        self.decoder = Decoder(self.embedding_layer,config) 
        self.discriminator = Discriminator(config) 


    def encoder_forward(self,*args,**kwargs):
        return self.encoder(*args,**kwargs) 
    def decoder_forward(self,*args,**kwargs):
        return self.decoder(*args,**kwargs)
    # Sample a latent vector with latent size
    def sample_latent(self,batch_size):
        return torch.randn(batch_size,self.config.latent_size,device=self.device)
    # Sample Next Outputs for a given input latent vector to the max length specified 
    def sample(self,batch_size,max_len=100):
        #TODO: Sample Decoder and get next inputs 
        pass 
    def interpolate_latent(self,batch_size,states):
        #TODO: Interpolate around the latent space from a starting vector in the space
        pass


            

