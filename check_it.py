# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 14:51:45 2019

@author: User
"""
import io

import torch

def load_checkpoint( check_path, model, optimizer):
    with open(check_path, 'rb') as f:
        buffer = io.BytesIO(f.read())
    checkpoint = torch.load(buffer)
    #checkpoint = torch.load(check_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])                
    epoch = checkpoint['epoch']
    prev_loss = checkpoint['prev_loss']
    plot_losses = prev_loss
    
    print("\n Checkpoint loaded !")
    print("\n Training resumes from epoch %d: ", epoch)
    print("\n Best training loss to continue training from: ", min(prev_loss))
    
    return model, optimizer, epoch, plot_losses