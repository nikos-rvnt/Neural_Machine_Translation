# -*- coding: utf-8 -*-
"""
 
"""

from __future__ import unicode_literals, print_function, division
import sys
import os
import random
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader

from Encoder_Decoder import Encoder, Attention_Decoder 
from preprocessing import preprocess_data, fix_vocabulary, fix_pairs, readDataset
import check_it


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")

SOS_token = 0
EOS_token = 1
PAD_token = 2
MAX_LENGTH = 100



    
def showPlot(data):
    
    fig = plt.figure()
    plt.plot(data)
    fig.suptitle(' Loss per 1000 train. iters ')
    plt.xlabel(' Training iterations ')
    plt.ylabel(' Loss ')
    fig.show()
        
    

if __name__ == '__main__':
 
    
    ###########################################################################
    # when executed on command line   
    # e.g. python -i gls_2_seq_model.py 800 4 0.25 10 True 32 general
    if len(sys.argv)>2:
 
        hidden_s = int(sys.argv[1]) # hidden size
        number_lay = int(sys.argv[2]) # number of layers
        drpout = float(sys.argv[3])  # dropout
        num_epoches = int(sys.argv[4])  # number of epochs
        chck = int(sys.argv[5]) # whether or not to load a checkpoint
        btch = int(sys.argv[6]) # batch size
        attention = sys.argv[7]  # attention type
         
    
    ###########################################################################
    # Data Input & Preprocessing
    lang2 = 'eng'
    lang1 = 'gls'
    
    
    wer_m, wer_v, rouge_lf, meteory = [], [], [], []
    bleu_ind1, bleu_ind2, bleu_ind3, bleu_ind4 = [], [], [], []
    bleu_cum2, bleu_cum3, bleu_cum4 = [], [], []
    gleu_s, gleu_crp, bleu_crp = [], [], []
    plotLossesAll,  plotLossesValAll, lossesAllVal_perEpoch = [], [], []
    
    # 5 times cross validate data
    for cvTime in range(5):
        
        # read and shuffle dataset
        print("\n Reading dataset...") 
        parallel = readDataset( lang1, lang2)
        random.shuffle(parallel)

        # split dataset in 5 segments, each time one of it will be val/test dataset
        parallelSegments = [ parallel[ time*17542:(time+1)*17542 ] for time in range(5) ]
            
        for cv in range(5):
            
            train_data = [ parallelSegments[tr] for tr in range(5) if tr!=cv ]
            train_data = [ train_data[ii][jj] for ii in range(4) for jj in range(len(train_data[ii])) ]
            val_data = parallelSegments[cv][:int(17542/2)]
            test_data = parallelSegments[cv][int(17542/2):]
            
            # training data
            src_lang, trg_lang, pairs_train, len_train, trainDF = preprocess_data( train_data, lang1, lang2, 'train', False)
            
            w2i_in = src_lang.word2index
            i2w_in = src_lang.index2word
            w2c_in = src_lang.word2count
            w2i_out = trg_lang.word2index
            i2w_out = trg_lang.index2word
            w2c_out = trg_lang.word2count
            
            unk_value = True
            pairs_train, src_lang, trg_lang = fix_vocabulary( pairs_train, src_lang, trg_lang, 'train', unk=unk_value)
            
            w2i_in = src_lang.word2index
            i2w_in = src_lang.index2word
            w2c_in = src_lang.word2count
            w2i_out = trg_lang.word2index
            i2w_out = trg_lang.index2word
            w2c_out = trg_lang.word2count

            # preprocess validation data
            src_lang_test, trg_lang_test, pairs_val, len_val, valDF = preprocess_data( val_data, lang1, lang2, 'val', False)
            pairs_val = fix_pairs(pairs_val)    
                
            # preprocess test data
            src_lang_test, trg_lang_test, pairs_test, len_test, testDF = preprocess_data( test_data, lang1, lang2, 'test', False)
            pairs_test = fix_pairs(pairs_test)            
            
            print("\n Fygame... ")
            
           
            ###########################################################################
            # Parameter Setting
            # e.g. num_layers=4 / hidden_size=800, num_layers=1 / hidden_size=180
            if len(sys.argv)>2:
                hidden_size = hidden_s
                num_layers = number_lay
                drpt = drpout
                max_len = 100
                attn = attention
                num_epochs = num_epoches
                check = chck
                btch_sz = btch
            else:
                hidden_size = 800
                num_layers = 4
                drpt = 0.25 
                max_len = 100
                attn = 'general' # one of 'general', 'dot', 'concat', 'bahdanau' attention mechanism
                num_epochs = 50
                check = True
                check_back = False
                btch_sz = 32
            learning_r = 0.001
            cvs = [ cvTime, cv]
            
            # Creating Encoder, Decoder Objects
            encoder = Encoder( src_lang.n_words, hidden_size, num_lay = num_layers, dropout=drpt).to(device)
            attnt_decoder = Attention_Decoder( attn, hidden_size, trg_lang.n_words, n_layers=num_layers, dropout=drpt).to(device)
            #encoder_optim = optim.Adam( encoder.parameters(), lr=0.00001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
            #decoder_optim = optim.Adam( attnt_decoder.parameters(), lr=0.00001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
            encoder_optim = optim.Adamax( encoder.parameters(), lr=learning_r, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
            decoder_optim = optim.Adamax( attnt_decoder.parameters(), lr=learning_r, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
        
        
            # create dir for checkpoints if not exists
            check_path = '/media/nikos/Data/didak/HealthSign/implement/gls2eng_attn_model/model_checkpoint/'
            if not os.path.exists(check_path):
                os.mkdir(check_path)
                
            # data to tensor
            import to_tensors
            pairs_train = to_tensors.variablesFromPairs(src_lang, trg_lang, pairs_train, MAX_LENGTH)
            pairs_train = DataLoader(pairs_train, batch_size=btch_sz, shuffle=True)
            pairs_val = to_tensors.variablesFromPairs(src_lang, trg_lang, pairs_val, MAX_LENGTH)
            pairs_val = DataLoader( pairs_val, batch_size=btch_sz, shuffle=True)
            
                
            ###########################################################################
            # Training Phase 
            import training
            plot_losses, plot_lossVal, lossVal_perEpoch, time_per_epoch, epochIndx = training.trainEpochs(encoder, attnt_decoder, encoder_optim, decoder_optim, 
                                              pairs_train, pairs_val, num_epochs, cvs, print_every=2000, 
                                              plot_every=2000, learning_rate=learning_r, check_pnt=True)
            
            times_path = '/media/nikos/Data/didak/HealthSign/implement/gls2eng_attn_model/times/'
            if not os.path.exists(times_path):
                os.mkdir(times_path)
            path_to_save_time = times_path + '_' + str(cvTime) + '_' + str(cv) + "time_per_epoch_" + lang1 + "2" + lang2 + "_" + attn + ".npy"
            np.save( path_to_save_time, time_per_epoch)
            #showPlot(plot_losses[0::100])
          
       
            ###########################################################################
            # Evaluation Phase - Computing translations
            print("\nTranslating test sentences...")
            pairs_test = to_tensors.variablesFromPairs(src_lang, trg_lang, pairs_test, MAX_LENGTH)
            pairs_test = DataLoader(pairs_test, batch_size=btch_sz, shuffle=True)
            
            del pairs_train
            del pairs_val
            
            import evaluation
            if os.path.exists(check_path):
                checkPointPath = check_path + 'attn_model_' + str(cvs[0]) + '_' + str(cvs[1]) + '_epoch_' + str(epochIndx) + '.pth'
                encoder, encoder_optimizer, attnt_decoder, decoder_optimizer, n_iters, plot_losses = check_it.load_checkpoint(checkPointPath, encoder, encoder_optim, attnt_decoder, decoder_optim)
                
            compare_results = []
            compare_results, eval_time = evaluation.evaluate_them( encoder, attnt_decoder, pairs_test, trg_lang, max_length=MAX_LENGTH)
            
            path_to_save_time = '/media/nikos/Data/didak/HealthSign/implement/gls2eng_attn_model/times/' + 'eval_time_per_sentence_'+lang1+"2"+lang2+"_"+str(num_layers)+"_"+attn+".npy"
            np.save( path_to_save_time, eval_time)
            
            comp_res = []
            temp_comp = compare_results
            for res32 in temp_comp:
                for i in range(len(res32)):
                    comp_res.append(res32[i])
        
            
            ###########################################################################
            # Computing WER and BLEU scores
            from score_translation import score_compute     
            print("\nComputing score metrics...")
            wer_mean, wer_var, bleus, gleus, meteor, rouge_l_f1 = score_compute( comp_res )
            
            bleu_indi = { 'bleu_indi1_mean': bleus[0], 'bleu_indi2_mean': bleus[1], 'bleu_indi3_mean': bleus[2], 'bleu_indi4_mean': bleus[3]}
            bleu_cum = { 'bleu_cum2_mean': bleus[4], 'bleu_cum3_mean': bleus[5], 'bleu_cum4_mean': bleus[6]}
            bleu_corpus = bleus[7]
            gleu_sentence = gleus[0]
            gleu_corpus = gleus[1]
            
            
            # for i in range(len(comp_res)):
                
                    
            #     if 'DESC' in comp_res[i][0]:
            #         comp_res[i][0] = comp_res[i][0].replace('DESC','DESC-')
                
            #     if 'X' in comp_res[i][0]:
            #         comp_res[i][0] = comp_res[i][0].replace('X','X-')
            
            #     if 'DESC' in comp_res[i][1]:
            #         comp_res[i][1] = comp_res[i][1].replace('DESC','DESC-')
                
            #     if 'X' in comp_res[i][1]:
            #         comp_res[i][1] = comp_res[i][1].replace('X','X-')
                    
            
            wer_m.append(wer_mean), wer_v.append(wer_var), rouge_lf.append(rouge_l_f1), meteory.append(meteor) 
            bleu_ind1.append(bleu_indi['bleu_indi1_mean']), bleu_ind2.append(bleu_indi['bleu_indi2_mean']), bleu_ind3.append(bleu_indi['bleu_indi3_mean']), bleu_ind4.append(bleu_indi['bleu_indi4_mean'])
            bleu_cum2.append(bleu_cum['bleu_cum2_mean']), bleu_cum3.append(bleu_cum['bleu_cum3_mean']), bleu_cum4.append(bleu_cum['bleu_cum4_mean'])
            gleu_s.append(gleu_sentence), gleu_crp.append(gleu_corpus) 
            bleu_crp.append(bleu_corpus)
    
    wer_meanf = np.mean(wer_m)
    wer_varf = np.mean(wer_v)
    rouge_lf1f = np.mean(rouge_lf)
    meteorf = np.mean(meteory)
    bleu_indi1f = np.mean(bleu_ind1)
    bleu_indi2f = np.mean(bleu_ind2)
    bleu_indi3f = np.mean(bleu_ind3)
    bleu_indi4f = np.mean(bleu_ind4)
    bleu_cum2f = np.mean(bleu_cum2)
    bleu_cum3f = np.mean(bleu_cum3)
    bleu_cum4f = np.mean(bleu_cum4)
    gleu_sf = np.mean(gleu_s)
    gleu_crpf = np.mean(gleu_crp)
    bleu_crpf = np.mean(bleu_crp)
    