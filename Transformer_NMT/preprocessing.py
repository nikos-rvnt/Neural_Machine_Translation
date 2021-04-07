# -*- coding: utf-8 -*-
"""
 
"""

from __future__ import unicode_literals, print_function, division
import torch
import pandas as pd
import io
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SOS_token = 0
EOS_token = 1
PAD_token = 2
MAX_LENGTH = 100


class Lang:
   def __init__( self, name):
       self.name = name
       self.word2index = {"UNK":0}
       self.word2count = {}
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
   '''        
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
   '''
            
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
        s = s.str.normalize('NFD')
        s = s.str.encode('ascii', errors='ignore').str.decode('utf-8') # ascii -> unicode

#def tokenizer(self, sentence):
#        sentence = re.sub(
#        r"[\*\"“”\n\\…\+\-\/\=\(\)‘•:\[\]\|’\!;]", " ", str(sentence))
#        sentence = re.sub(r"[ ]+", " ", sentence)
#        sentence = re.sub(r"\!+", "!", sentence)
#        sentence = re.sub(r"\,+", ",", sentence)
#        sentence = re.sub(r"\?+", "?", sentence)
#        sentence = sentence.lower()
#        return [tok.text for tok in self.nlp.tokenizer(sentence) if tok.text != " "]
#        
    return s


def read_sentence(df, lang1, lang2):
   sentence1 = normalizeSentence( df, lang1)
   sentence2 = normalizeSentence( df, lang2)
   return sentence1, sentence2

def read_file( loc1, lang1, lang2):
   df1 = pd.read_csv(loc1, delimiter='\t', header=None, names=[lang1, lang2])
   #df2 = pd.read_csv(loc2, delimiter='\t', header=None, names=[lang1, lang2])
   length_train = len(df1)
   #length_test = len(df2)
   df = pd.concat([df1])
   
   return df, length_train


def readDataset( lang1, lang2):

    path = '/media/nikos/Data/Datasets/ASLG-PC12/'
    annos_path =  path + 'aslg-pc12.csv'
    
    # read dataset from .csv
    annos = pd.read_csv( annos_path, sep=';', header=None, names = ['eng','gls'])
    eng_anno = annos['eng'] 
    gls_anno = annos['gls']        
    
    parallel = []
    if lang1 == 'gls':
        for i in range(len(annos)):
            parallel.append( [gls_anno[i], eng_anno[i]])
    else:
        for i in range(len(annos)):
            parallel.append( [eng_anno[i], gls_anno[i]])        
    
    return parallel





def preprocess_data( parData, lang1, lang2, phase, rvrsd = False):
        
   # if phase == 'train':
   #     if lang1 == 'gls':
   #         data_loc = path + 'gls2eng_ds_train.txt'
   #     elif lang1 == 'eng':
   #         data_loc = path + 'eng2gls_ds_train.txt'
   # else:
   #     if lang1 == 'gls':
   #         data_loc = path + 'gls2eng_ds_test.txt'
   #     elif lang1 == 'eng':
   #         data_loc = path + 'eng2gls_ds_test.txt'
   
   # from list to dataframe
   if lang1 == 'gls':
       paralDF = pd.DataFrame( parData, columns = [ 'gls', 'eng'])
   elif lang1 == 'eng':
       paralDF = pd.DataFrame( parData, columns = [ 'eng', 'gls'])
    
   if phase == 'train':
       print("\n Normalising training data...") 
   elif phase == 'val':
       print("\n Normalising validation data...") 
   elif phase == 'test':
       print("\n Normalising test data...") 
   
   sentenceIn, sentenceOut = read_sentence( paralDF, lang1, lang2)
   sentenceIn = sentenceIn.tolist()
   sentenceOut = sentenceOut.tolist()

   source = Lang(lang1)
   target = Lang(lang2)
   
   pairs = []
   length = len(paralDF)
   for i in range( length ):
       if len(sentenceIn[i].split(' ')) < MAX_LENGTH and len(sentenceOut[i].split(' ')) < MAX_LENGTH:
           # if input chosen to be in a reversed order to reduce vanishing gradient phenomenon
           if rvrsd:
               temp = sentenceIn[i].split(' ')
               sentenceIn[i] = ' '.join(list(reversed(temp)))
           # only training data to vocabulary
           if phase == 'train':
               source.addSentence(sentenceIn[i])
               target.addSentence(sentenceOut[i])
               
           both = [sentenceIn[i], sentenceOut[i]]
           pairs.append(both)

   return source, target, pairs, length, paralDF


def fix_vocabulary( pairs_train, src_voc, trg_voc, phase, unk=True):
    
    
    
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
        
    unk_repl = unk
    if unk_repl:    
        # replace every unknown word (previously deleted from dict_in, dict_out) with 'UNK'
        for i in range( len( pairs_train ) ):
            
            list0 = pairs_train[i][0].split(' ')
            for j in range( len( list0 )):
                if list0[j] in src_voc.word2count:
                    # reducing input dictionary of counts per word. If word count < 5 go away
                    if src_voc.word2count[list0[j]] < 4:
                        list0[j] = 'UNK' 
                        src_voc.n_words -= 1
                if list0[j] not in src_voc.word2count and list0[j] != ' ':
                    list0[j] = 'UNK'
                    
            pairs_train[i][0] = ' '.join( word for word in list0)
                    
            list1 = pairs_train[i][1].split(' ')
            for z in range( len( list1 )):
                if list1[z] in trg_voc.word2count:
                    # reducing output dictionary of counts per word. If word count < 5 go away
                    if trg_voc.word2count[list1[z]] < 4:
                        list1[z] = 'UNK' 
                        trg_voc.n_words -= 1
                if list1[z] not in trg_voc.word2count and list1[z] != ' ':
                    list1[z] = 'UNK'
            
            pairs_train[i][1] = ' '.join( word for word in list1)
         
            
        # src_voc.addWord('UNK')
        # trg_voc.addWord('UNK')
        
        lang1 = src_voc.name
        lang2 = trg_voc.name
        
        del src_voc
        del trg_voc
        
        # recreate src_lang and trg_lang input dictionairies
        src_voc = Lang(lang1)
        trg_voc = Lang(lang2)
        src_voc.addWord('UNK')
        trg_voc.addWord('UNK')
                        
        for pair in pairs_train:
             
            src_voc.addSentence(pair[0])
            trg_voc.addSentence(pair[1])
        
    
    return pairs_train, src_voc, trg_voc    


def fix_pairs(pairs_test):
    
    # remove exra spaces
    for i in range( len(pairs_test) ):
        tempIn = pairs_test[i][0].split(' ')
        tempIn1 = []
        for word in tempIn:
            if word != '' and word != ' ':
                tempIn1.append(word)
        pairs_test[i][0] = ' '.join( word11 for word11 in tempIn1)
        
        tempOut = pairs_test[i][1].split(' ')
        tempOut1 = []
        for word2 in tempOut:
            if word2 != '' and word2 != ' ':
                tempOut1.append(word2)
        pairs_test[i][1] = ' '.join( word22 for word22 in tempOut1)
    
    return pairs_test

