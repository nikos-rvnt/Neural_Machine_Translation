# -*- coding: utf-8 -*-
"""
 
"""

from __future__ import unicode_literals, print_function, division

import os, sys, stat
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from Transformer_Classes import Transformer_Model
from preprocessing import preprocess_data, fix_vocabulary, fix_pairs, readDataset
import check_it


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")

SOS_token = 0
EOS_token = 1
PAD_token = 2
MAX_LENGTH = 60



    
def showPlot(data):
    
    fig = plt.figure()
    plt.plot(data)
    fig.suptitle(' Loss per 1000 train. iters ')
    plt.xlabel(' Training iterations ')
    plt.ylabel(' Loss ')
    fig.show()
        
    

if __name__ == '__main__':
 
    torch.set_num_threads(4)
    ###########################################################################
    # when executed on command line   
    # e.g. python -i transformer_main.py 800 4 0.25 10 True 32 general
    if len(sys.argv)>2:
 
        mdl_name = int(sys.argv[1]) # model_name: 'base' or 'big' transformer
        num_epoches = int(sys.argv[2])  # 'base': 12, 'big': 25
        chck = int(sys.argv[3]) # whether or not to load a checkpoint
        btch = int(sys.argv[4]) # batch size
         
    
    ###########################################################################
    # Data Input & Preprocessing
    lang1 = 'gls'
    lang2 = 'eng'
    
    wer_m, wer_v, rouge_lf, meteory = [], [], [], []
    bleu_ind1, bleu_ind2, bleu_ind3, bleu_ind4 = [], [], [], []
    bleu_cum2, bleu_cum3, bleu_cum4 = [], [], []
    gleu_s, gleu_crp, bleu_crp = [], [], []
    plotLossesAll,  plotLossesValAll, lossesAllVal_perEpoch = [], [], []
    
    path_train = '/media/nikos/Data/Datasets/phoenix-2014-T.v3/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/annotations/manual/PHOENIX-2014-T.train.corpus.csv'
    path_dev = '/media/nikos/Data/Datasets/phoenix-2014-T.v3/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/annotations/manual/PHOENIX-2014-T.dev.corpus.csv'
    path_test = '/media/nikos/Data/Datasets/phoenix-2014-T.v3/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/annotations/manual/PHOENIX-2014-T.test.corpus.csv'
    
    # 5 times cross validate data
    for cvTime in range(5):
        
        # # read and shuffle dataset
        # print("\n Reading dataset...") 
        # parallel = readDataset( lang1, lang2)
        # random.shuffle(parallel)

        # # split dataset in 5 segments, each time one of it will be val/test dataset
        # parallelSegments = [ parallel[ time*17542:(time+1)*17542 ] for time in range(5) ]
            
        for cv in range(1):
            
            # train_data = [ parallelSegments[tr] for tr in range(5) if tr!=cv ]
            # train_data = [ train_data[ii][jj] for ii in range(4) for jj in range(len(train_data[ii])) ]
            # val_data = parallelSegments[cv][:int(17542/2)]
            # test_data = parallelSegments[cv][int(17542/2):]
            

            train_data = pd.read_csv( path_train, sep='|',usecols=['orth','translation']).values.tolist()
            val_data = pd.read_csv( path_dev, sep='|',usecols=['orth','translation']).values.tolist()
            test_data = pd.read_csv( path_test, sep='|',usecols=['orth','translation']).values.tolist()

            
            # training data
            src_lang, trg_lang, pairs_train, len_train, trainDF = preprocess_data( train_data, lang1, lang2, 'train', False)
            
            w2i_in = src_lang.word2index
            i2w_in = src_lang.index2word
            w2c_in = src_lang.word2count
            w2i_out = trg_lang.word2index
            i2w_out = trg_lang.index2word
            w2c_out = trg_lang.word2count
            
            unk_value = False
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
                ntokens_in = src_lang.n_words  # size of input vocabulary
                ntokens_out = trg_lang.n_words # size of output vocabulary
                drop_it = 0.2
                embed_size = 512    # embedding dimension
            else:
                ntokens_in = src_lang.n_words  # size of input vocabulary
                ntokens_out = trg_lang.n_words # size of output vocabulary
                embed_size = 1024   # embedding dimension 512 for base // 1024 for big
                drop_it = 0.3
            
            ###########################################################################
            # Training Phase parameters
            if len(sys.argv)>2:
                num_epochs = num_epoches
                check = chck
                btch_sz = btch
                model_name = mdl_name
            else:
                num_epochs = 65
                check = False
                check_back = False
                btch_sz = 2
                model_name = 'big'
            learning_r = 0.00001
            cvs = [ cvTime, cv]
            
        #    count_in = 0
        #    count_out = 0
        #    for pair in pairs_test:
        #        for word in pair[0].split(' '):
        #            if word not in src_lang.word2index:
        #                count_in += 1
        #        for word1 in pair[1].split(' '):
        #            if word1 not in trg_lang.word2index:
        #                count_out += 1    
        #    
        #    print("Tosa agnwsta gloss: \n", count_in)
        #    print("Toses agnwstes lekseis: \n", count_out)
            
            import to_tensors
            pairs_train = to_tensors.variablesFromPairs(src_lang, trg_lang, pairs_train, MAX_LENGTH)
            pairs_train = DataLoader(pairs_train, batch_size=btch_sz, shuffle=True)
            pairs_val = to_tensors.variablesFromPairs(src_lang, trg_lang, pairs_val, MAX_LENGTH)
            pairs_val = DataLoader( pairs_val, batch_size=btch_sz, shuffle=True)
            
            ###########################################################################
            # Transformer
            model = Transformer_Model( ntokens_in, ntokens_out, embed_size, drop_it, trns_model=model_name).to(device)
            
            # Adamax optimizer
            optimizer = optim.Adam( model.parameters(), lr=learning_r, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
        
            # create dir for checkpoints if not exists
            check_path = '/media/nikos/Data/didak/HealthSign/implement/gls2eng_transformer/model_checkpoint/'
            if not os.path.exists(check_path):
                os.mkdir(check_path)
                
            ###########################################################################
            # Training Phase 
            import training
            plot_losses, plot_lossVal, lossVal_perEpoch, time_per_epoch, epochIndx = training.train_it(model, pairs_train, pairs_val, num_epochs, 
                                                           optimizer, btch_sz, cvs, print_every=100, check_pnt=True, continueFromCheck = False)
             
            
            plotLossesAll.append(plot_losses)
            plotLossesValAll.append(plot_lossVal)
            lossesAllVal_perEpoch.append(lossVal_perEpoch)

            times_path = '/media/nikos/Data/didak/HealthSign/implement/gls2eng_transformer/times/'
            if not os.path.exists(times_path):
                os.mkdir(times_path)
            path_to_save_time = times_path + '_' + str(cvTime) + '_' + str(cv) + "time_per_epoch_"+lang1+"2"+lang2+"_"+model_name+".npy"
            np.save( path_to_save_time, time_per_epoch)
        #    showPlot(plot_losses[0::100])    
            if os.path.exists(check_path):
                checkPointPath = check_path + 'transformer_checkpoint_' + str(cvs[0]) + '_' + str(cvs[1]) + '_epoch_' + str(epochIndx) + '.pth'
                model, optimizer, n_iters, plot_losses = check_it.load_checkpoint( checkPointPath, model, optimizer)
             
               
            ###########################################################################
            # Evaluation Phase - Computing translations    
            print("\nTranslating test sentences...")
            del pairs_train
            del pairs_val
            pairs_test = to_tensors.variablesFromPairs(src_lang, trg_lang, pairs_test, MAX_LENGTH)
            pairs_test = DataLoader(pairs_test, batch_size=btch_sz, shuffle=True) 
            
            import evaluation
            compare_results = []
            compare_results, eval_time = evaluation.evaluate_it( model, pairs_test, trg_lang, btch_sz, max_length=MAX_LENGTH)
            
            path_to_save_time = '/media/nikos/Data/didak/HealthSign/implement/gls2eng_transformer/times/eval_time_per_sentence_'+lang1+"2"+lang2+"_"+model_name+".npy"
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
            gleu_sentense = gleus[0]
            gleu_corpus = gleus[1]
            
            wer_m.append(wer_mean), wer_v.append(wer_var), rouge_lf.append(rouge_l_f1), meteory.append(meteor) 
            bleu_ind1.append(bleu_indi['bleu_indi1_mean']), bleu_ind2.append(bleu_indi['bleu_indi2_mean']), bleu_ind3.append(bleu_indi['bleu_indi3_mean']), bleu_ind4.append(bleu_indi['bleu_indi4_mean'])
            bleu_cum2.append(bleu_cum['bleu_cum2_mean']), bleu_cum3.append(bleu_cum['bleu_cum3_mean']), bleu_cum4.append(bleu_cum['bleu_cum4_mean'])
            gleu_s.append(gleu_sentense), gleu_crp.append(gleu_corpus) 
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
  