# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 11:57:00 2019

@author: User
"""

import torch


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SOS_token = 0
EOS_token = 1
PAD_token = 2

def indexesFromSentence(lang, sentence):
    
    #return [lang.word2index[word] for word in sentence.split(' ') ]
    return [lang.word2index[word]  if word in lang.word2index else lang.word2index['UNK'] for word in sentence.split(' ') ]


def variableFromSentence(lang, sentence, max_length=100):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    indexes.extend([PAD_token] * (max_length - len(indexes)))
    result = torch.LongTensor(indexes)
    if device.type=="cuda":
        return result.cuda()
    else:
        return result

def variablesFromPairs(input_lang, output_lang, pairs, max_length):
    res = []
    for pair in pairs:
        input_variable = variableFromSentence(input_lang, pair[0], max_length)
        target_variable = variableFromSentence(output_lang, pair[1], max_length)
        res.append((input_variable, target_variable))
    return res