# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 12:09:23 2019

@author: User
"""

# -*- coding: utf-8 -*-
"""

"""

import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional  
from torch.autograd import Variable

import random
import numpy as np
import time
import math
from datetime import datetime

import to_tensors
import check_it
import evaluation


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")

SOS_token = 0
EOS_token = 1
MAX_LENGTH = 100

 
###############################################################################
#    
# helper functions to print time elapsed and estimated 
# time remaining given the current time and progress 
#    
###############################################################################
def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

'''
    Implement Label Smoothing Regularization
'''
class LabelSmoothing(nn.Module):
    
    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(size_average=False) # Kullback–Leibler divergence / relative entropy
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None
        
    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, Variable(true_dist, requires_grad=False))


def train_it( model, train_data, val_data, epochs, optimizer, batch_s, cvs, print_every=100, check_pnt=False, continueFromCheck = False):
    
    print("\n Training has started...")
    
    model.train()    
    criterion = nn.CrossEntropyLoss()
    #criterion = LabelSmoothing( size=6900, padding_idx=0, smoothing=0.1)
    
    # to load a checkpoint and resume training from that epoch
    epochi = 0
    # checkpoint path
    check_path = '/media/nikos/Data/didak/HealthSign/implement/gls2eng_transformer/model_checkpoint/' 
    if continueFromCheck:
        chk_point = os.listdir(check_path)
        if os.path.isfile(chk_point[-1]):
            model, optimizer, epoch, plot_losses = check_it.load_checkpoint( chk_point[-1], model, optimizer)
            n_iters = epochs - epoch
            print("\n Training continues from epoch: ", n_iters)    
            epochi = epoch+1

    length_train = len(train_data)
    length_val = len(val_data)
    plot_losses = []    
    #plot_lossVal = []  
    lossVal_perEpoch = []  
    lossesVal_PerEpoch = []     
    time_train = []
    for epoch in range( epochi, epochs):
        
        model.train()  
        
        start_epoch = time.time()
        total_loss = 0
        print_loss_total = 0
        plot_loss_total = 0
        train_count = 0    
        loss = 0
        
        
        for batch_x, batch_y in train_data: 
            #batch_x = batch_x.to("cpu")
            #batch_y = batch_y.to("cpu")
                        
            cur_loss = 0
                        
            # right shift target input, inserting SOS_token in the first place
            target_input1 = Variable(torch.LongTensor([SOS_token] * batch_x.size(0)))
            target_input1 = target_input1.cuda() if device.type=="cuda" else target_input1
            #target_input1 = target_input1.to("cpu")
            batch_y_train = torch.cat(( target_input1.unsqueeze(1), batch_y[ :, :-1]), dim=1)

            out_pred = model( batch_x.transpose(0,1), batch_y_train.transpose(0,1))
            target_length = batch_y.size()[1]
            for di in range( target_length ):
                loss += criterion( out_pred[di], batch_y.transpose(0,1)[di])
            
                 
            # calculate gradient
            loss.backward()
            # gradient clipping both for encoder & decoder
            torch.nn.utils.clip_grad_value_( model.parameters(), 0.5)
            # update parameters
            optimizer.step()
            # clear gradient of all tensors
            optimizer.zero_grad()
            
            cur_loss = loss.item() / target_length
            total_loss += cur_loss
            print_loss_total += cur_loss 
            plot_loss_total += cur_loss
            loss = loss.item() / target_length
                                   
            print("\nTraining epoch: ",epoch+1,"/",epochs)
            print("Batch Number: ",train_count,"/",length_train)
            print("Batch training loss: ", loss)
            
            if (train_count+1) % print_every == 0:
                print_loss_avg = print_loss_total / print_every
                print_loss_total = 0
                print('-----  %s (%d %d%%) %.4f -------' % (timeSince(start_epoch, 
                                    (train_count+1) / length_train), (train_count+1), 
                                    (train_count+1)/(len(train_data)/32)*100, print_loss_avg))
    
            train_count += 1
            plot_losses.append(loss)
        
        time_epoch = time.time() - start_epoch  
        time_train.append(time_epoch)
        
        print("\n ---- Epoch %d needed: [%.3s] seconds ---- \n" %(epoch, timeSince(start_epoch, 
                                                                (train_count+1) / length_train)))
        
        #### - Validation -
        model.eval()  
        
        start_epoch = time.time()
        total_loss_val = 0
        print_loss_val_total = 0
        plot_loss_val_total = 0
        val_count = 0    
        loss_val = 0
        plot_lossVal = []  

        
        for batch_valX, batch_valY in val_data: 
            #batch_x = batch_x.to("cpu")
            #batch_y = batch_y.to("cpu")
                        
            cur_loss = 0
                        
            # right shift target input, inserting SOS_token in the first place
            target_input2 = Variable(torch.LongTensor([SOS_token] * batch_valX.size(0)))
            target_input2 = target_input2.cuda() if device.type=="cuda" else target_input2
            #target_input1 = target_input1.to("cpu")
            batch_y_val = torch.cat(( target_input2.unsqueeze(1), batch_valY[ :, :-1]), dim=1)

            out_predVal = model( batch_valX.transpose(0,1), batch_y_val.transpose(0,1))
            target_length = batch_valY.size()[1]
            for di in range( target_length ):
                loss_val += criterion( out_predVal[di], batch_valY.transpose(0,1)[di])
            
                 
            # calculate gradient
            # loss_val.backward()
            # # gradient clipping both for encoder & decoder
            # torch.nn.utils.clip_grad_value_( model.parameters(), 0.5)
            # # update parameters
            # optimizer.step()
            # # clear gradient of all tensors
            # optimizer.zero_grad()
            
            cur_loss_val = loss_val.item() / target_length
            total_loss_val += cur_loss_val
            print_loss_val_total += cur_loss_val 
            plot_loss_val_total += cur_loss_val
            loss_val = loss_val.item() / target_length
                                   
            print("\nValidation epoch: ",epoch+1,"/",epochs)
            print("Batch Number: ",val_count,"/",length_val)
            print("Batch validation loss: ", loss_val)
            
            if (train_count+1) % print_every == 0:
                print_loss_val_avg = print_loss_val_total / print_every
                print_loss_val_total = 0
                print('-----  %s (%d %d%%) %.4f -------' % (timeSince(start_epoch, 
                                    (train_count+1) / length_train), (train_count+1), 
                                    (train_count+1)/(len(val_data)/32)*100, print_loss_val_avg))
    
            val_count += 1
            plot_lossVal.append(loss_val)
            
            # #if epoch%5==0 and epoch>0:
            # if len(plot_lossVal) >= 5:
            #     indx_min_loss = np.argmin(plot_lossVal[-10:])
            #     if indx_min_loss in [4,5,6]:
                    
            #         pths = os.listdir(check_path)
            #         for chk in range( len( pths )):
            #             if str(indx_min_loss)+".pth" not in pths[chk]: 
            #                 os.remove(check_path + pths[chk])
                    
            #         print('Training epoch ' + str(indx_min_loss) + ' found to be the one with the minimum loss and training phase should stop!')
            #         return plot_losses, plot_lossVal, lossVal_perEpoch, time_train
                        
        lossVal_perEpoch.append(np.mean(plot_lossVal))
        lossesVal_PerEpoch.append(plot_lossVal)

        # save a checkpoint every epoch
        if check_pnt:
                #checkPth = '/media/nikos/Data/didak/HealthSign/implement/gls2eng_transformer/model_checkpoint/transformer_checkpoint' + now.strftime("_%m:%d:%Y_%H:%M:%S_") + 'epoch_' + str(epoch) + '.pth'
                checkPth = '/media/nikos/Data/didak/HealthSign/implement/gls2eng_transformer/model_checkpoint/transformer_checkpoint_' + str(cvs[0]) + '_' + str(cvs[1]) + '_epoch_' + str(epoch) + '.pth'

                check_pnt = checkPth  
                #train_iter = int( epoch / len(train_data) )
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss,
                    'prev_loss': plot_losses,
                    'epoch': epoch,
                    }, check_pnt)
                print("\n Checkpoint saved!")

        # #if epoch%5==0 and epoch>0:
        if (epoch-epochi) >= 5:
            indx_min_loss = np.argmin(lossVal_perEpoch)
            if indx_min_loss<(len(lossVal_perEpoch)-2):
                
                pths = os.listdir(check_path)
                for chk in range( len( pths )):
                    if str(indx_min_loss)+".pth" not in pths[chk]:
                            str2chk = '_' + str(cvs[0]) + '_' + str(cvs[1]) + '_epoch_'
                            if str2chk in pths[chk]:
                                os.remove(check_path + pths[chk])
                
                print('Training epoch ' + str(indx_min_loss) + ' found to be the one with the minimum loss and training phase should stop!')
                return plot_losses, lossesVal_PerEpoch, lossVal_perEpoch, time_train, indx_min_loss
    
    return plot_losses, lossesVal_PerEpoch, lossVal_perEpoch, time_train , epoch     

