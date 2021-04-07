# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 14:51:45 2019

@author: User
"""

import io
import torch

def load_checkpoint( check_path, encoder, encoder_optimizer, decoder, decoder_optimizer):
    
    with open(check_path, 'rb') as f:
        buffer = io.BytesIO(f.read())
    checkpoint = torch.load(buffer)  
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    encoder_optimizer.load_state_dict(checkpoint['encoder_optimizer_state_dict'])
    decoder.load_state_dict(checkpoint['decoder_state_dict'])
    decoder_optimizer.load_state_dict(checkpoint['decoder_optimizer_state_dict'])                
    epoch = checkpoint['epoch']
    prev_loss = checkpoint['prev_loss']
    plot_losses = prev_loss
    
    print("\n Checkpoint loaded !")
    print("\n Training resumes from epoch %d: ", epoch)
    print("\n Best training loss to contine training from %f: \n", min(prev_loss))
    
    return encoder, encoder_optimizer,decoder, decoder_optimizer, epoch, plot_losses