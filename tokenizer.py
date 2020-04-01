#!/bin/bash/python 
# Author: Ragy Haddad <ragy@deepbiometherapeutics.com>
import sys,os
import pandas as pd 
import numpy as np 
from config import Config 
import torch  

# Tokenizer class for encoding and decoding utils 
class Tokenizer:
    def __init__(self):
        self.config = Config() 
        self.char_to_int = self.config.char_to_int 
        self.int_to_char = self.config.int_to_char 
        self.max_len = self.config.max_len   
    # 2D Logits to String
    def encoding_to_string(self,decoder_output_batch):
        int_arr_ = []
        str_out = []
        for m in decoder_output_batch:
            matrix = m 
            max_idx = torch.argmax(matrix).item()
            int_arr_.append(max_idx) 
            str_out.append(self.int_to_char[str(max_idx)])  
        str_out = "".join(str_out)
        return str_out
    # Smiles to 2D 
    def smiles_to_2d(self,smiles_string,padding=False,add_start_token=False):
        def encode(smiles_string):
            arr_ = np.zeros((len(smiles_string),self.config.vocab_size),dtype=np.float32)
            for idx,c in enumerate(smiles_string):
                arr_[idx][self.char_to_int[c]] = 1 
            return arr_  
        if padding:
            if add_start_token:
                smiles_string = self.config.sos_token + smiles_string + (self.max_len - len(smiles_string) - 1) * self.config.eos_token
            else:
                smiles_string = smiles_string  + (self.max_len - len(smiles_string)) * self.config.eos_token 
        else:
            if add_start_token:
                smiles_string = self.config.sos_token + smiles_string + self.config.eos_token
            else:
                smiles_string = smiles_string  + self.config.eos_token 
        return encode(smiles_string) 
    # Smiles to 1D Int
    def smiles_to_int(self,smiles_string,padding=False,add_start_token=True):
        def encode(smiles_string):
            arr_ = np.zeros(len(smiles_string),dtype=np.int32) 
            for idx,c in enumerate(smiles_string):
                arr_[idx] = self.char_to_int[c] 
            return arr_ 
        if padding:
            if add_start_token:
                smiles_string = self.config.sos_token + smiles_string + (self.max_len - len(smiles_string) - 1) * self.config.eos_token
            else:
                smiles_string = smiles_string + (self.max_len - len(smiles_string) - 1) * self.config.eos_token  
        else:
            if add_start_token:
                smiles_string = self.config.sos_token + smiles_string + self.config.eos_token
            else:
                smiles_string = smiles_string + self.config.eos_token  
        return encode(smiles_string) 

    # 1D Int to Smiles
    def int_to_smiles(self,int_arr):
        return  ".".join([self.int_to_char[i] for i in int_arr])

    









