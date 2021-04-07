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
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

def evaluate_them(encoder, decoder, dataloader, output_lang, max_length=100):

    
    encoder.eval()
    decoder.eval()
    
    compare_res = []
    start_time = time.time()
    for batch_x, batch_y in dataloader:

        batch_size = batch_x.size()[0]
        encoder_hidden = encoder.initHidden(batch_size)

        input_variable = Variable(batch_x.transpose(0, 1))
        target_variable = Variable(batch_y.transpose(0, 1))

        input_length = input_variable.size()[0]
        target_length = target_variable.size()[0]

        output = torch.LongTensor(target_length, batch_size)

        encoder_outputs = Variable(torch.zeros(max_length, batch_size, encoder.hidden_size))
        encoder_outputs = encoder_outputs.cuda() if device.type=="cuda" else encoder_outputs
        
        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(
                input_variable[ei], batch_size, encoder_hidden)
            encoder_outputs[ei] = encoder_output[0]

        decoder_input = Variable(torch.LongTensor([SOS_token] * batch_size))
        decoder_input = decoder_input.cuda() if device.type=="cuda" else decoder_input
        decoder_hidden = encoder_hidden

        for di in range(target_length):
            decoder_output, decoder_hidden = decoder( decoder_input, batch_size, decoder_hidden, encoder_outputs)
            topv, topi = decoder_output.data.topk(1)

            output[di] = torch.cat( tuple(topi) )
            decoder_input = torch.cat( tuple(topi) ) 

        output = output.transpose(0,1)
        compare_batch = []
        for di in range(output.size()[0]):
            
            translate_di = []
            for i in range( len(output[di]) ):
                if output[di][i] not in [0,1,2]:
                    if output[di][i].item() in output_lang.index2word:
                        translate_di.append( output_lang.index2word[ output[di][i].item() ] )
                    else:
                        translate_di.append('UNK')
            trans_temp = ' '.join(translate_di)
            
            original_di = []
            for i in range( len(batch_y[di]) ):
                if batch_y[di][i] not in [0,1,2]:
                    original_di.append( output_lang.index2word[ batch_y[di][i].item() ] )
            orig_temp = ' '.join(original_di)
            
            batch_duo = [orig_temp , trans_temp]
            compare_batch.append(batch_duo)
                       
    
        compare_res.append(compare_batch)
        
    eval_time = time.time() - start_time
    eval_time = eval_time / (len(dataloader)*batch_size)
    return compare_res, eval_time




