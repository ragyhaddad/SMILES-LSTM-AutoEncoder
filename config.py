#!/bin/bash/python 
# Author: Ragy Haddad <ragy@deepbiometherapeutics.com>
import sys,os

# Global Configuration For Model Training 

class Config:
    def __init__(self):
        self.latent_size = 32 
        self.hidden_size = 128
        self.char_to_int = {"@": 0, "s": 1, "!": 2, "[": 3, "H": 4, "P": 5, "7": 6, "i": 7, "#": 8, "1": 9, "=": 10, "2": 11, "S": 12, "/": 13, "E": 14, "-": 15, "c": 16, "n": 17, "+": 18, ")": 19, "N": 20, "]": 21, "5": 22, "3": 23, "o": 24, "4": 25, "C": 26, "O": 27, "l": 28, "\\": 29, "I": 30, "6": 31, "(": 32, "B": 33, "F": 34, "r": 35} 
        self.int_to_char = {"0": "@", "1": "s", "2": "!", "3": "[", "4": "H", "5": "P", "6": "7", "7": "i", "8": "#", "9": "1", "10": "=", "11": "2", "12": "S", "13": "/", "14": "E", "15": "-", "16": "c", "17": "n", "18": "+", "19": ")", "20": "N", "21": "]", "22": "5", "23": "3", "24": "o", "25": "4", "26": "C", "27": "O", "28": "l", "29": "\\", "30": "I", "31": "6", "32": "(", "33": "B", "34": "F", "35": "r"}
        self.vocab_size = len(self.int_to_char)
        self.embedding_size = len(self.int_to_char)  
        self.sos_token = '!' 
        self.eos_token = 'E'
        self.num_layers = 4
        self.padding_int_value = self.char_to_int[self.eos_token] # Padding Int Value 
        self.sos_token_int = self.char_to_int[self.sos_token] 
        self.discriminator_layers = [640, 256]
        self.device = 'cuda:0'
        self.lr = 0.001 
        self.bidirectional = True 
        self.max_len = 128 
        self.epochs = 10 
        self.batch_size = 128


    
