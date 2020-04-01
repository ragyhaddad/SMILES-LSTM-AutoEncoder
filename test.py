#!/bin/bash/python
# Author: Ragy Haddad <ragy@deepbiometherapeutics.com> 
import sys,os 
import numpy as np 
from model import AAE 
from dataloader import Dataset 
from config import Config 
import torch 
from tokenizer import Tokenizer
import torch.nn.functional as F  
import numpy as np 
from rdkit import Chem 
from rdkit.Chem import Draw 
from rdkit.Chem import inchi  
 

# Load AAE 
def load_model(model_path):
    config = Config()
    device = torch.device(config.device) 
    model = AAE(config).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model 

# Test Reconstruction Loss Only
def test_autoencoder(model_path,test_path):
    config = Config()
    device = torch.device(config.device) 
    model = AAE(config).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval() 
    dataloader = Dataset(test_path,sep=',') 
    tokenizer = Tokenizer()
    total = dataloader.num_smiles()
    correct = 0
    batch_size = 100
    # Test Dataset
    with torch.no_grad():
        for idx,data in enumerate(dataloader.load_batches(batch_size=batch_size,label_encoding=True,shifted_targets=True,padding=True,return_length=True)):
            x,y,l_x,l_y = data   # src,target,src_length,trg_length
            x,y = torch.from_numpy(x).long().to(device),torch.from_numpy(y).long().to(device)
            l_x,l_y = torch.from_numpy(l_x).to(device),torch.from_numpy(l_y).to(device)
            encoder_inputs = (x,l_x) 
            latent_codes = model.encoder_forward(x,l_x) 
            smiles = latent_to_smiles(model,latent_codes) 
            for idx,smi in enumerate(smiles):
                y_str = [config.int_to_char[str(i.item())] for i in y[idx]]
                y_str = "".join(y_str).replace('E','')
                if smi == y_str:
                    correct += 1
            print(correct/total)
# Latent Vector To Smiles
def latent_to_smiles(model,latent_codes,remove_eos=True):
    config = Config()
    device = torch.device(config.device)
    decoder_inputs_init = torch.zeros(latent_codes.shape[0],128).to(device).long()
    lengths = torch.full((latent_codes.shape[0],),fill_value=config.max_len).long() # Set Max To Max Length
    decoder_inputs = (decoder_inputs_init,lengths)
    decoder_outputs, decoder_output_lengths, _ = model.decoder_forward(
                        *decoder_inputs, latent_codes, is_latent_states=True) 
    
    batch_size = decoder_outputs.shape[0]
    smiles = []
    for b in range(batch_size):
        d_str = []
        for i in decoder_outputs[b]:
            m_idx = torch.argmax(i).item()
            c = config.int_to_char[str(m_idx)] 
            # Clip at EOS Char
            if remove_eos and c == 'E': 
                break
            d_str.append(c)
        d_str = "".join(d_str) 
        smiles.append(d_str) 
    return smiles

def smiles_to_latent(model,smiles):
    config = Config()
    tokenizer = Tokenizer()
    device = torch.device(config.device)
    smi = tokenizer.smiles_to_int(smiles,padding=True,add_start_token=True)
    smi = torch.tensor([smi]).long().to(device) 
    lengths = torch.full((1,),fill_value= len(smiles) + 2 ).long().to(device)
    latent_codes = model.encoder_forward(smi,lengths)
    return latent_codes
# Diversify a given molecule using latent space interpolation 
def interpolate_latent_space(model_path,randomize=False):
    config = Config()
    model = load_model(model_path) 
    device = torch.device(config.device)
    mols = []
    smi_i = 'NCCC1=ONCC1' 
    smi_j = 'CC1=NCCCON1=C'
    l0 = smiles_to_latent(model,smi_i) 
    l1 = smiles_to_latent(model,smi_j)
    mols.append(Chem.MolFromSmiles(smi_i))
    # Linear Interpolate
    ratios = np.linspace(0,1,25) 
    scale = 0.4 # If testing random interpolation
    added = []
    for idx,r in enumerate(ratios):
        if randomize == False:
            rlatent = (1.0 - r) * l0 + r * l1
        else: 
            rlatent = l0 + torch.from_numpy(scale*(np.random.randn(l0.shape[0]))).to(device).float()
        s = latent_to_smiles(model,rlatent)
        m = Chem.MolFromSmiles(s)
        if m is not None:
            inchi_key = inchi.MolToInchiKey(m)
            if m not in mols and inchi_key not in added: 
                mols.append(m) 
                added.append(inchi_key)
    mols.append(Chem.MolFromSmiles(smi_j))
    img = Draw.MolsToGridImage(mols, molsPerRow=4)
    img.save('figures/interpolate.png')

def main(): 
    # interpolate_latent_space(sys.argv[1],randomize=True)
    test_autoencoder(sys.argv[1],sys.argv[2])

if __name__ == "__main__":
    main()