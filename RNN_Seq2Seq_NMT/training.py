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
import torch.nn.functional as F
from torch.autograd import Variable

import random
import numpy as np

import to_tensors
import check_it

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SOS_token = 0
EOS_token = 1
MAX_LENGTH = 100



######################################################################
# Training the Model
# ------------------
#
# To train we run the input sentence through the encoder, and keep track
# of every output and the latest hidden state. Then the decoder is given
# the ``<SOS>`` token as its first input, and the last hidden state of the
# encoder as its first hidden state.
#
# "Teacher forcing" is the concept of using the real target outputs as
# each next input, instead of using the decoder's guess as the next input.
# Using teacher forcing causes it to converge faster but `when the trained
# network is exploited, it may exhibit
# instability <http://minds.jacobs-university.de/sites/default/files/uploads/papers/ESNTutorialRev.pdf>`__.
#
# You can observe outputs of teacher-forced networks that read with
# coherent grammar but wander far from the correct translation -
# intuitively it has learned to represent the output grammar and can "pick
# up" the meaning once the teacher tells it the first few words, but it
# has not properly learned how to create the sentence from the translation
# in the first place.
#
# Because of the freedom PyTorch's autograd gives us, we can randomly
# choose to use teacher forcing or not with a simple if statement. Turn
# ``teacher_forcing_ratio`` up to use more of it.
######################################################################
teacher_forcing_ratio = 0.5


def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=MAX_LENGTH):
    
    batch_size = input_tensor.size()[0]
    encoder_hidden = encoder.initHidden(batch_size)

    input_tensor = Variable(input_tensor.transpose(0, 1))
    target_tensor = Variable(target_tensor.transpose(0, 1))

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size()[0]
    target_length = target_tensor.size()[0]

    encoder_outputs = Variable(torch.zeros(max_length, batch_size, encoder.hidden_size))
    encoder_outputs = encoder_outputs.cuda() if device.type=="cuda" else encoder_outputs

    loss = 0
    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(
            input_tensor[ei], batch_size, encoder_hidden)
        encoder_outputs[ei] = encoder_output[0]

    decoder_input = Variable(torch.LongTensor([SOS_token] * batch_size))
    decoder_input = decoder_input.cuda() if device.type=="cuda" else decoder_input

    decoder_hidden = encoder_hidden

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
     

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
#            decoder_output, decoder_hidden, decoder_attention = decoder(
#                decoder_input, batch_size, decoder_hidden, encoder_outputs)
            decoder_output, decoder_hidden = decoder( decoder_input, batch_size, decoder_hidden, encoder_outputs)
            loss += criterion(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di]  # Teacher forcing
            
    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden = decoder( decoder_input, batch_size, decoder_hidden, encoder_outputs) 
            topv, topi = decoder_output.data.topk(1)
            decoder_input = Variable(torch.cat( tuple( topi ) )) 
            # decoder_input = Variable(torch.LongTensor([[ni]]))
            decoder_input = decoder_input.cuda() if device.type=="cuda" else decoder_input
    
            loss += criterion(decoder_output, target_tensor[di]) 

    # calculate gradient
    loss.backward()

    # gradient clipping both for encoder & decoder
    torch.nn.utils.clip_grad_value_( encoder.parameters(), 5)
    torch.nn.utils.clip_grad_value_( decoder.parameters(), 5)

    # update parameters
    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length


def validate(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=MAX_LENGTH):
    
    batch_size = input_tensor.size()[0]
    encoder_hidden = encoder.initHidden(batch_size)

    input_tensor = Variable(input_tensor.transpose(0, 1))
    target_tensor = Variable(target_tensor.transpose(0, 1))
    input_length = input_tensor.size()[0]
    target_length = target_tensor.size()[0]

    encoder_outputs = Variable(torch.zeros(max_length, batch_size, encoder.hidden_size))
    encoder_outputs = encoder_outputs.cuda() if device.type=="cuda" else encoder_outputs

    loss = 0
    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(
            input_tensor[ei], batch_size, encoder_hidden)
        encoder_outputs[ei] = encoder_output[0]

    decoder_input = Variable(torch.LongTensor([SOS_token] * batch_size))
    decoder_input = decoder_input.cuda() if device.type=="cuda" else decoder_input

    decoder_hidden = encoder_hidden

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
     

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):

            decoder_output, decoder_hidden = decoder( decoder_input, batch_size, decoder_hidden, encoder_outputs)
            decoder_input = target_tensor[di]  # Teacher forcing
            loss += criterion(decoder_output, target_tensor[di])
            
    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            
            decoder_output, decoder_hidden = decoder( decoder_input, batch_size, decoder_hidden, encoder_outputs) 
            topv, topi = decoder_output.data.topk(1)
            decoder_input = Variable(torch.cat( tuple( topi ) )) 
            decoder_input = decoder_input.cuda() if device.type=="cuda" else decoder_input
            loss += criterion(decoder_output, target_tensor[di]) 


    return loss.item() / target_length

###############################################################################
#    
# helper functions to print time elapsed and estimated time
# remaining given the current time and progress 
#    
###############################################################################
import time
import math

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

    

###############################################################################
def trainEpochs(encoder, decoder, encoder_optimizer, decoder_optimizer, train_data, val_data, epochs, cvs, print_every=1000, plot_every=100, learning_rate=0.01, check_pnt = False, continueFromCheck = False):
    
    print("\n Training has started...")
      
    
    criterion = nn.CrossEntropyLoss()
    
    # to load a checkpoint and resume training from that epoch
    epochi = 0
 
    if continueFromCheck:
        # checkpoint path
        check_path = '/media/nikos/Data/didak/HealthSign/implement/gls2eng_attn_model/model_checkpoint/'  
        if os.path.isfile(check_path):
            encoder, encoder_optimizer,decoder, decoder_optimizer, epoch, plot_losses = check_it.load_checkpoint(check_path, encoder, encoder_optimizer, decoder, decoder_optimizer)
            n_iters = epochs - epoch
            print("\n Training continues from epoch: %d", n_iters)    
            epochi = epoch+1
                
    train_count = 0
    val_count = 0
    time_train = []
    plot_losses = []
    plot_lossesVal = []
    len_train = len(train_data)
    len_val = len(val_data)
    plot_losses = []    
    #plot_lossVal = []  
    lossVal_perEpoch = []  
    lossesVal_PerEpoch = [] 
    
    for epoch in range( epochi, epochs):

        encoder.train()
        decoder.train()
        
        start = time.time()    
        train_count = 0 
        # training phase
        print_loss_total = 0  # Reset every print_every
        plot_loss_total = 0  # Reset every plot_every
        train_count = 0    
        for batch_x, batch_y in train_data:
            train_count +=1
            
            loss = train(batch_x, batch_y, encoder,
                         decoder, encoder_optimizer, decoder_optimizer, criterion, MAX_LENGTH) 
            print_loss_total += loss
            plot_loss_total += loss
            
            print("\nTraining epoch: ",epoch+1,"/",epochs)
            print("Batch Number: ",train_count-1,"/", len_train)
            print("Batch training loss: ", loss)
            if (train_count+1) % print_every == 0:
                print_loss_avg = print_loss_total / print_every
                print_loss_total = 0
                print('%s (%d %d%%) %.4f' % (timeSince(start, (train_count+1) / epochs),
                                             (train_count+1), (train_count+1) / (len(train_data)/32) * 100, print_loss_avg))
            plot_losses.append(loss)

        time_epoch = time.time() - start 
        time_train.append(time_epoch)

        encoder.eval()
        decoder.eval()           
            
        # validation phase
        val_count = 0
        plot_lossVal = []
        print_loss_totalVal = 0  # Reset every print_every
        plot_loss_totalVal = 0  # Reset every plot_every 
        for batch_valX, batch_valY in val_data:     
            val_count +=1
            
            lossVal = validate(batch_x, batch_y, encoder,
                         decoder, encoder_optimizer, decoder_optimizer, criterion, MAX_LENGTH) 
            print_loss_totalVal += lossVal
            plot_loss_totalVal += lossVal
            
            print("\nValidation epoch: ", epoch+1, "/", epochs)
            print("Batch Number: ", val_count-1, "/", len_val)
            print("Batch validation loss: ", lossVal)
            if (val_count+1) % print_every == 0:
                print_loss_avgVal = print_loss_totalVal / print_every
                print_loss_totalVal = 0
                print('%s (%d %d%%) %.4f' % (timeSince(start, (val_count+1) / epochs),
                                             (val_count+1), (val_count+1) / (len(val_data)/32) * 100, print_loss_avgVal))

            plot_lossVal.append(lossVal)
        lossVal_perEpoch.append(np.mean(plot_lossVal))
        lossesVal_PerEpoch.append(plot_lossVal)
  
        
        # save a checkpoint every epoch
        if check_pnt:
                checkPth = '/media/nikos/Data/didak/HealthSign/implement/gls2eng_attn_model/model_checkpoint/attn_model_'  + str(cvs[0]) + '_' + str(cvs[1]) + '_epoch_' + str(epoch) + '.pth'
                check_pth = checkPth 
            
                #train_iter = int( epoch / len(train_data) )
                torch.save({
                    'encoder_state_dict': encoder.state_dict(),
                    'decoder_state_dict': decoder.state_dict(),
                    'encoder_optimizer_state_dict': encoder_optimizer.state_dict(),
                    'decoder_optimizer_state_dict': decoder_optimizer.state_dict(),  
                    'loss': loss,
                    'prev_loss': plot_losses,
                    'epoch': epoch,
                    }, check_pth)
                print("\n Checkpoint saved!")
            
        check_path = '/media/nikos/Data/didak/HealthSign/implement/gls2eng_attn_model/model_checkpoint/'
        if (epoch-epochi) >= 5:
            indx_min_loss = np.argmin(lossVal_perEpoch)
            if indx_min_loss<(len(lossVal_perEpoch)-2) and np.min(lossVal_perEpoch)<0.08:
                
                pths = os.listdir(check_path)
                for chk in range( len( pths )):
                    if str(indx_min_loss)+".pth" not in pths[chk]:
                            str2chk = '_' + str(cvs[0]) + '_' + str(cvs[1]) + '_epoch_'
                            if str2chk in pths[chk]:
                                os.remove(check_path + pths[chk])
                
                print('Training epoch ' + str(indx_min_loss) + ' found to be the one with the minimum loss and training phase should stop!')
                return plot_losses, lossesVal_PerEpoch, lossVal_perEpoch, time_train, indx_min_loss
            
            
    return plot_losses, lossesVal_PerEpoch, lossVal_perEpoch, time_train, indx_min_loss

