# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 12:24:04 2019

@author: User
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 13:43:06 2019

@author: User
"""

import torch 
from torch.autograd import Variable
import numpy as np
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")

SOS_token = 0
EOS_token = 1
MAX_LENGTH = 100

import to_tensors

######################################################################
# Evaluation
# ==========
#
# Evaluation is mostly the same as training, but there are no targets so
# Wefeed the decoder's predictions back to itself for each step.
# Every time it predicts a word we add it to the output string, and if it
# predicts the EOS token we stop there. We also store the decoder's
# attention outputs for display later. 
######################################################################



def greedy_translation( model, batch_in, batch_out, output_lang, batch_s, max_length=100):
    
    init_inp = np.zeros( [ 1, batch_s], dtype=np.int64) # fill 1st batch input with SOS_token, that is zeros
    #decoder_input = Variable(torch.ones(1, 32).fill_( SOS_token )).type_as(torch.long)
    decoder_input = Variable(torch.LongTensor( init_inp )) # 1x1 SOS input
    #decoder_input = Variable(torch.LongTensor([SOS_token] * batch_in.transpose(0,1).size()))
    decoder_input = decoder_input.cuda() if device.type=="cuda" else decoder_input
    target_variable = Variable( batch_out.transpose(0, 1))
    target_length = target_variable.size()[0]
    #output = torch.LongTensor(target_length, batch_s)
    output = torch.zeros( [ target_length, batch_s], dtype=torch.long)
    
    compare_batch = []
    # compute encoder memory and use it as input to the decoder 
    memory = model.forward_eval_enc( batch_in.transpose(0,1) )
    for di in range(target_length):
        #print("\n To di: \n", di)
        
        # compute/decode each word using memory and previous output
        #out_pred = model.decoder( memory, decoder_input)        
        out_pred = model.forward_eval_dec( decoder_input, memory)
        topv, topi = out_pred.data.topk(1)
        
        output[di]  = torch.cat( tuple(topi[di]) ).squeeze()
        temp = torch.cat( tuple(topi[di]) ).squeeze().unsqueeze(0)
        decoder_input = torch.cat( (decoder_input, temp), dim=0 )

    output = output.transpose(0,1)
    
    for di in range(output.size()[0]):
        
        translate_di = []
        for i in range( len(output[di]) ):
            if output[di][i] not in [0,1,2]:
                translate_di.append( output_lang.index2word[ output[di][i].item() ] )
        trans_temp = ' '.join(translate_di)
        
        original_di = []
        for i in range( len(batch_out[di]) ):
            if batch_out[di][i] not in [0,1,2]:
                original_di.append( output_lang.index2word[ batch_out[di][i].item() ] )
        orig_temp = ' '.join(original_di)
        
        batch_duo = [orig_temp , trans_temp]
        compare_batch.append(batch_duo)
            
    return compare_batch
    
def beam_search_decode( model, batch_in, batch_out, output_lang, batch_s, beam_w = 3, max_length=100):
    
    init_inp = np.zeros( [ 1, batch_s], dtype=np.int64) # fill 1st batch input with SOS_token, that is zeros
    #decoder_input = Variable(torch.ones(1, 32).fill_( SOS_token )).type_as(torch.long)
    decoder_input = Variable(torch.LongTensor( init_inp )) # 1x1 SOS input
    #decoder_input = Variable(torch.LongTensor([SOS_token] * batch_in.transpose(0,1).size()))
    decoder_input = decoder_input.cuda() if device.type=="cuda" else decoder_input
    target_variable = Variable( batch_out.transpose(0, 1))
    target_length = target_variable.size()[0]
    #output = torch.LongTensor(target_length, batch_s)
    output = torch.zeros( [ target_length, batch_s], dtype=torch.long)
    
    compare_batch = []
    # compute encoder memory and use it as input to the decoder 
    memory = model.forward_eval_enc( batch_in.transpose(0,1) )
    
    for di in range(batch_s):
        
        # compute/decode each word using memory and previous output
        #out_pred = model.decoder( memory, decoder_input)        
        out_pred = model.forward_eval_dec( decoder_input, memory)
        topv, topi = out_pred.data.topk(beam_w)
        
        #output[di] = torch.cat( tuple(topi[di]) ).squeeze()
        temp = torch.cat( tuple(topi[di]) ).squeeze().unsqueeze(0)
        decoder_input = torch.cat( (decoder_input, temp), dim=0 )
        
        holdProbs = list()
        holdProbs_Ind = list()
        
        while True:
            
            for prob in holdProbs:
                                
                # compute/decode each word using memory and previous output
                #out_pred = model.decoder( memory, decoder_input)        
                out_pred = model.forward_eval_dec( decoder_input, memory)
                topv, topi = out_pred.data.topk(beam_w)
                
                output[di] = torch.cat( tuple(topi[di]) ).squeeze()
                temp = torch.cat( tuple(topi[di]) ).squeeze().unsqueeze(0)
                decoder_input = torch.cat( (decoder_input, temp), dim=0 )
    
    
    

def evaluate_it( model, test_data, output_lang, batch_s, max_length=100):

    model.eval()
    
    compare_results = []
    i=0
    start_time = time.time()
    for batch_x, batch_y in test_data:
#        batch_x = batch_x.to("cpu")
#        batch_y = batch_y.to("cpu")
        i += 1
        print("\n Batch_Num: ", i)
        #print(" Batch Size: \n", batch_x.size()[0])
        
        comp_batch = greedy_translation( model, batch_x, batch_y, output_lang, batch_x.size()[0])
        compare_results.append(comp_batch)
    
    eval_time = time.time() - start_time
    eval_time = eval_time / (len(test_data)*batch_s)
    
    return compare_results, eval_time


