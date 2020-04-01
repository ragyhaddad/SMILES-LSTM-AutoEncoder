#!/bin/bash/python 
# Author: Ragy Haddad <ragy@deepbiometherapeutics.com>
import sys,os
import pandas as pd 
import numpy as np
from tokenizer import Tokenizer 
from torch.utils.data import Dataset as torchDataset
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader as torchDataLoader
from torch.nn.utils.rnn import pad_sequence 
import torch
from rdkit import Chem 
# Dataset functionality for custom batch loading
class Dataset:
    def __init__(self,input_table,sep='\t'):
        self.df = pd.read_csv(input_table,sep=sep) 
        
        self.t = Tokenizer() 
        if 'SMILES' in self.df.columns:
            self.smiles = self.df['SMILES'] 
        elif 'smiles' in self.df.columns:
            self.smiles = self.df['smiles']
        elif 'Smiles' in self.df.columns:
            self.smiles = self.df['Smiles']
        else:
            self.smiles = self.df.iloc[:,0] # Column Zero is smiles if none of the above  
    
        self.cans = [Chem.MolToSmiles(Chem.MolFromSmiles(smi),True) for smi in self.smiles]
        self.smiles = pd.DataFrame(self.cans).iloc[:,0]
        # print(self.smiles)
        # exit()
    # Format Batch 
    def format_df(self,input_df,return_labels=False,shifted_targets=True,padding=False,label_encoding=False,return_length=True):
        data = None 
        if shifted_targets and label_encoding == False:
            src = [self.t.smiles_to_2d(r,padding=padding,add_start_token=True) for r in input_df] 
            trg = [self.t.smiles_to_2d(r,padding=padding,add_start_token=False) for r in input_df] 
        if shifted_targets and label_encoding:
            src = [self.t.smiles_to_int(r,padding=padding,add_start_token=True) for r in input_df]
            trg = [self.t.smiles_to_int(r,padding=padding,add_start_token=False) for r in input_df] 
        if return_labels:
            labels = [self.t.smiles_to_int(r,padding=padding,add_start_token=False) for r in input_df] # Labels do not have starting token  
            src,trg,labels = np.array(src),np.array(trg),np.array(labels)
            data = (src,trg,labels) 
        else:
            src,trg = np.array(src),np.array(trg)
            data = (src,trg) 
        if return_length:
            l_src = np.array([len(x) + 2 for x in input_df]) # Add 2 to lengths for sos_token and end_token
            l_trg = np.array([len(x) + 1 for x in input_df]) # Add 1 to lengths for eos_token  
            data = data + (l_src,l_trg)
        return data 
    # Load Batches
    def load_batches(self,batch_size,epoch_shuffle=True,padding=False,return_labels=False,label_encoding=False,shifted_targets=True,return_length=False):
        if epoch_shuffle:
            self.smiles = self.smiles.sample(frac=1)
        i = 0 
        while i < len(self.df.index.values):
            smiles = self.smiles.iloc[i : i + batch_size]
            
            data = self.format_df(smiles,padding=padding,return_labels=return_labels,label_encoding=label_encoding,return_length=return_length)  
            i += batch_size 
            yield data 
    
    def num_smiles(self):
        return len(self.smiles)


        
    
