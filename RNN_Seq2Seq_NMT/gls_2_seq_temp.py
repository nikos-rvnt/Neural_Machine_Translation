# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 17:34:09 2019

@author: User
"""

# -*- coding: utf-8 -*-
"""
 
"""

from __future__ import unicode_literals, print_function, division
import torch
import torch.optim as optim

import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

import sys
import os

import check_it
from Encoder_Decoder import Encoder, Attention_Decoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SOS_token = 0
EOS_token = 1
PAD_token = 2
MAX_LENGTH = 100


class Lang:
   def __init__( self, name):
       self.name = name
       self.word2index = { }
       self.word2count = { }
       self.index2word = {0: "SOS", 1: "EOS", 2:"PAD", 3: "UNK"}
       self.n_words = 4  # Count SOS, EOS, PAD, UNK
       self.count = 0
       
   def addSentence(self, sentence):
           #split a sentence and add each word
           for word in sentence.split(' '):
               self.addWord(word)
 
   def addWord(self, word):
       if word not in self.word2index:
           self.word2index[word] = self.n_words
           self.word2count[word] = 1
           self.index2word[self.n_words] = word
           self.n_words += 1
       else:
           self.word2count[word] += 1

   def addSentenceTest(self, sentence, lang_obj):
           for word in sentence.split(' '):
               self.addWordTest(word, lang_obj)
           
   def addWordTest(self, word, lang_obj):
        
        if (word in lang_obj.word2index) and (word not in self.word2index):
            self.word2index[word] = lang_obj.word2index[word]
            self.word2count[word] = lang_obj.word2count[word]
            self.index2word[lang_obj.word2index[word]] = word
            #self.n_words += 1
        elif (word not in lang_obj.word2index) and (word not in self.word2index):
            self.word2index["UNK"] = lang_obj.word2index["UNK"]
            self.word2count["UNK"] = lang_obj.word2count["UNK"]
            self.word2count["UNK"] += 1
            self.index2word[lang_obj.word2index["UNK"]] = "UNK" 
            self.n_words += 1
            self.count += 1
            

def normalizeSentence(df, lang):
 
    if lang=='eng':    
        s = df[lang].str.lower() # to lower case
        s = s.str.replace( '([!?])', ' \1 ')
        s = s.str.replace( '\s[.]', ' ')
        s = s.str.replace( '[^A-Za-z\s]+', '') 
        s = s.str.normalize('NFD')
        s = s.str.encode('ascii', errors='ignore').str.decode('utf-8') # ascii -> unicode
    else:
        s = df[lang] 
        s = s.str.replace( '([!?])', ' \1 ')
        s = s.str.replace( '\s[.]', ' ')
        s = s.str.replace( '[^A-Za-z\s]+', '') 
        #s = s.str.normalize('NFD')
        s = s.str.encode('ascii', errors='ignore').str.decode('utf-8') # ascii -> unicode
        
    return s


def read_sentence(df, lang1, lang2):
   sentence1 = normalizeSentence(df, lang1)
   sentence2 = normalizeSentence(df, lang2)
   return sentence1, sentence2

def read_file(loc1, lang1, lang2):
   df1 = pd.read_csv(loc1, delimiter='\t', header=None, names=[lang1, lang2])
   #df2 = pd.read_csv(loc2, delimiter='\t', header=None, names=[lang1, lang2])
   length_train = len(df1)
   #length_test = len(df2)
   df = pd.concat([df1])
   
   return df, length_train

def preprocess_data(lang1,lang2, src_obj, trg_obj, phase, rvrsd = False):
    
   path = 'D:\\Datasets\\ASLG-PC12\\'
   if phase == 'train':
       data_loc = path + 'gls2eng_ds_train.txt'
   else:
       data_loc = path + 'gls2eng_ds_test.txt'
   
#   path = 'D:\\Datasets\\phoenix-2014-T.v3\\PHOENIX-2014-T-release-v3\\PHOENIX-2014-T\\annotations\\manual\\'
#   train_loc = path + 'gls2ger_ds_train.txt'
#   test_loc = path + 'gls2ger_ds_test.txt'
    
   df, length = read_file( data_loc, lang1, lang2) 
   print("\n Reading %s input-output pairs..." % (length)) 
   sentenceIn, sentenceOut = read_sentence(df, lang1, lang2)
   sentenceIn = sentenceIn.tolist()
   sentenceOut = sentenceOut.tolist()

   if phase == 'test': 
       src_test = Lang(lang1)
       trg_test = Lang(lang2)
   
   pairs = []
   for i in range(len(df)):
       if len(sentenceIn[i].split(' ')) < MAX_LENGTH and len(sentenceOut[i].split(' ')) < MAX_LENGTH:
           # if input chosen to be in a reversed order to reduce vanishing gradient phenomenon
           if rvrsd:
               temp = sentenceIn[i].split(' ')
               sentenceIn[i] = ' '.join(list(reversed(temp)))
           #
           if phase == "train":
               src_obj.addSentence(sentenceIn[i])
               trg_obj.addSentence(sentenceOut[i])
           else:
               src_test.addSentenceTest(sentenceIn[i], src_obj)
               trg_test.addSentenceTest(sentenceOut[i], trg_obj)
           both = [sentenceIn[i], sentenceOut[i]]
           pairs.append(both)
           
   if phase == 'train':
       source = src_obj
       target = trg_obj
   else:
       source = src_test
       target = trg_test

   return source, target, pairs, length, df


    
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
    # e.g. python -i seq_2_seq_model.py 800 4 0.25 10 True 32 general
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
    lang1 = 'gls'
    lang2 = 'eng'
    src_train = Lang(lang1)
    trg_train = Lang(lang2)
    src_lang, trg_lang, pairs_train, len_train, df = preprocess_data(lang1, lang2, src_train, trg_train, 'train', False)
    
    w2i_in = src_lang.word2index
    i2w_in = src_lang.index2word
    w2c_in = src_lang.word2count
    w2i_out = trg_lang.word2index
    i2w_out = trg_lang.index2word
    w2c_out = trg_lang.word2count
    
    
    # fixing pairs - remove \ufeff from first sentence of file]
#    pairs_train[0][0] = pairs_train[0][0].replace('\ufeff', '', 1)
#    pairs[len_train][0] = pairs[len_train][0].replace('\ufeff', '', 1)
#
#    pairs_conc = [ pairs[i][0] + ' ' + pairs[i][1] for i in range(len(pairs))  ]
#    
#    pairs_t = []
#    for i in range( 70168, 87710):
#        count = 0
#        for j in range( 0, 70168 ):
#            
#            if pairs_conc[i]!=pairs_conc[j]:
#                pairs_t.append(pairs_conc[i])
#                count +=1
#                break
#    pairs_test = (list( set(pairs_t) ))
#    
#    pairs = (list(set(pairs_conc)) )
#    pairs_fxd = []
#    for i in range( len(pairs) ):
#        
#        for j in range( len(pairs[i]) ):
#            
#            if pairs[i][j].islower():
#                break
#            
#        pairs_fxd.append([ pairs[i][:j-1], pairs[i][j:]] )
##        print("\n Gls: " + pairs[i][:j-1] + " \n")
##        print("\n Eng: " + pairs[i][j:] + " \n\n")
#        
#    
#    for i in range( len(pairs_fxd) ):
#        
#        if pairs_fxd[i] == ' ':
#            continue
#        if len(pairs_fxd[i][0]) > 1:
#            if pairs_fxd[i][0][-1]==' ':
#                pairs_fxd[i][0] = pairs_fxd[i][0][:-1]
#        
#        if len(pairs_fxd[i][1]) > 1:
#            if pairs_fxd[i][1][-1]==' ':
#                pairs_fxd[i][1] = pairs_fxd[i][1][:-1]     
#    pairs = pairs_fxd           
    
    
    unk_repl = True
    if unk_repl:
        for i in range( len(pairs_train) ):
            
            tempIn = pairs_train[i][0].split(' ')
            tempIn1 = []
            for word in tempIn:
                if word != '':
                    tempIn1.append(word)
            pairs_train[i][0] = ' '.join( word11 for word11 in tempIn1)
            
            tempOut = pairs_train[i][1].split(' ')
            tempOut1 = []
            for word2 in tempOut:
                if word2 != '':
                    tempOut1.append(word2)
            pairs_train[i][1] = ' '.join( word22 for word22 in tempOut1)
    
        # reducing input dictionary of counts per word. If word count < 5 go away
        cnt = 0
        to_delete = []
        for word in src_lang.word2count:
             
            if src_lang.word2count[word]<5 or word=='' and (word not in ['SOS','EOS']):
                cnt += src_lang.word2count[word]
                to_delete.append(word)
                    
        for d in to_delete:
            temp = src_lang.word2index[d]
            del src_lang.word2count[d]
            del src_lang.word2index[d]
            del src_lang.index2word[temp]
            src_lang.n_words -= 1
        
        
        # reducing output dictionary of counts per word. If word count < 5 go away
        cnt = 0
        to_delete2 = []
        for word2 in trg_lang.word2count:
             
            if trg_lang.word2count[word2]<5 or word2=='' and  (word2 not in ['SOS','EOS']):
                cnt += trg_lang.word2count[word2]
                to_delete2.append(word2)                
        
        for d2 in to_delete2:
            temp2 = trg_lang.word2index[d2]
            del trg_lang.word2count[d2]
            del trg_lang.word2index[d2]
            del trg_lang.index2word[temp2]
            trg_lang.n_words -= 1
            
        
            
        # replace every unknown word (previously deleted from dict_in, dict_out) with 'UNK'
        for i in range( len( pairs_train ) ):
            
            list0 = pairs_train[i][0].split(' ')
            for j in range( len( list0 )):
                if list0[j] not in w2c_in and list0[j] != ' ':
                    list0[j] = 'UNK'
                    
            pairs_train[i][0] = ' '.join( word for word in list0)
                    
            list1 = pairs_train[i][1].split(' ')
            for z in range( len( list1 )):
                
                if list1[z] not in w2c_out and list1[z] != ' ':
                    list1[z] = 'UNK'
            
            pairs_train[i][1] = ' '.join( word for word in list1)
         
            
        src_lang.addWord('UNK')
        trg_lang.addWord('UNK')
        
        del src_lang
        del trg_lang
        
        # recreate src_lang and trg_lang input dictionairies
        src_lang = Lang(lang1)
        trg_lang = Lang(lang2)
        
        for pair in pairs_train:
             
            src_lang.addSentence(pair[0])
            trg_lang.addSentence(pair[1])
        
        w2i_in = src_lang.word2index
        i2w_in = src_lang.index2word
        w2c_in = src_lang.word2count
        w2i_out = trg_lang.word2index
        i2w_out = trg_lang.index2word
        w2c_out = trg_lang.word2count
    
    # test data
    src_lang_test, trg_lang_test, pairs_test, len_test, df = preprocess_data(lang1, lang2, src_lang, trg_lang, 'test', False)
    
    for i in range( len(pairs_test) ):
        tempIn = pairs_test[i][0].split(' ')
        tempIn1 = []
        for word in tempIn:
            if word != '':
                tempIn1.append(word)
        pairs_test[i][0] = ' '.join( word11 for word11 in tempIn1)
        
        tempOut = pairs_test[i][1].split(' ')
        tempOut1 = []
        for word2 in tempOut:
            if word2 != '':
                tempOut1.append(word2)
        pairs_test[i][1] = ' '.join( word22 for word22 in tempOut1)
        
        
    # replace every unknown word (previously deleted from dict_in, dict_out) with 'UNK'
    for i in range( len( pairs_test ) ):
        
        list0 = pairs_test[i][0].split(' ')
        for j in range( len( list0 )):
            if list0[j] not in w2c_in and list0[j] != ' ':
                list0[j] = 'UNK'
                
        pairs_test[i][0] = ' '.join( word for word in list0)
                
        list1 = pairs_test[i][1].split(' ')
        for z in range( len( list1 )):
            
            if list1[z] not in w2c_out and list1[z] != ' ':
                list1[z] = 'UNK'
        
        pairs_test[i][1] = ' '.join( word for word in list1)
    
    # recreate src_lang and trg_lang input dictionairies
    src_test_lang = Lang(lang1)
    trg_test_lang = Lang(lang2)
    
    for pair in pairs_test:
         
        src_test_lang.addSentenceTest( pair[0], src_lang)
        trg_test_lang.addSentenceTest( pair[1], trg_lang)
      
    w2i_test_in = src_lang_test.word2index
    i2w_test_in = src_lang_test.index2word
    w2c_test_in = src_lang_test.word2count
    w2i_test_out = trg_lang_test.word2index
    i2w_test_out = trg_lang_test.index2word
    w2c_test_out = trg_lang_test.word2count

#    print(src_lang_test.count)
#    print(trg_lang_test.count)
 

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
    else:
        hidden_size = 350
        num_layers = 2
        drpt = 0.25 
        max_len = 100
        attn = 'dot' # one of 'general', 'dot', 'concat', 'bahdanau' attention mechanism
    
    
    ###########################################################################
    # Creating Encoder, Decoder Objects
    encoder = Encoder( src_lang.n_words, hidden_size, num_lay = num_layers).to(device)
    attnt_decoder = Attention_Decoder( attn, hidden_size, trg_lang.n_words, n_layers=num_layers, dropout=drpt).to(device)

    
    ###########################################################################
    # Training Phase 
    if len(sys.argv)>2:
        num_epochs = num_epoches
        check = chck
        btch_sz = btch
    else:
        num_epochs = 5
        check = False
        check_back = False
        btch_sz = 32
    learning_r = 0.001
     #len_train = 10

    #encoder_optim = optim.Adam( encoder.parameters(), lr=0.00001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
    #decoder_optim = optim.Adam( attnt_decoder.parameters(), lr=0.00001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
    encoder_optim = optim.Adamax( encoder.parameters(), lr=learning_r, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
    decoder_optim = optim.Adamax( attnt_decoder.parameters(), lr=learning_r, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

    # create dir for checkpoints if not exists
    check_path = "D:\\Datasets\\ASLG-PC12\\model_checkpoint\\checkpoint.pth"
    if check_back and os.path.exists(check_back):
        os.remove(check_path)
    if not os.path.exists(check_path.rsplit('checkpoint.pth',1)[0]):
        os.mkdir("D:\\Datasets\\ASLG-PC12\\model_checkpoint\\")
    
    import to_tensors
    pairs_train = to_tensors.variablesFromPairs(src_lang, trg_lang, pairs_train, MAX_LENGTH)
    pairs_train = DataLoader(pairs_train, batch_size=btch_sz, shuffle=True)
    
    import training
    plot_losses, time_per_epoch = training.trainEpochs(encoder, attnt_decoder, encoder_optim, decoder_optim, 
                                      pairs_train, epochs=num_epochs, print_every=2000, 
                                      plot_every=2000, learning_rate=learning_r, check_pnt=check)
        
    if os.path.exists(check_path):
        encoder, encoder_optimizer, attnt_decoder, decoder_optimizer, n_iters, plot_losses = check_it.load_checkpoint(check_path, encoder, encoder_optim, attnt_decoder, decoder_optim)
    
   
    showPlot(plot_losses[0::100])
    
    
    ###########################################################################
    # Evaluation Phase - Computing translations
    print("\nTranslating test sentences...")
    pairs_test = to_tensors.variablesFromPairs(src_lang_test, trg_lang_test, pairs_test, MAX_LENGTH)
    pairs_test = DataLoader(pairs_test, batch_size=btch_sz, shuffle=True)
    
    import evaluation
    compare_results = []
    compare_results = evaluation.evaluate_them( encoder, attnt_decoder, pairs_test, trg_lang_test, max_length=MAX_LENGTH)
    
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
    
 