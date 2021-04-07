# -*- coding: utf-8 -*-

"""

 Function to calculate 
 
 WER, BLEU score, GLEU score, METEOR, ROUGE-L F1 score

"""


# BLEU:
# https://machinelearningmastery.com/calculate-bleu-score-for-text-python/
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu
from nltk.translate.gleu_score import sentence_gleu, corpus_gleu
from nltk.translate.meteor_score import single_meteor_score

from find_WER import wer
from rouge import Rouge
import numpy as np
#import sacrebleu

def score_compute( comp_res ):
    
    res_wer = []
    bleu_indi1 = []
    bleu_indi2 = []
    bleu_indi3 = []
    bleu_indi4 = []
    bleu_cum2 = []
    bleu_cum3 = []
    bleu_cum4 = []
    gleu_sent = []
    meteor_score = []
    rouge_score = []
     
    translated = []
    reference = []
    for i in range( len(comp_res) ):
        reference.append([comp_res[i][0].split(' ')])
        translated.append(comp_res[i][1].split(' '))
    bleu_corpus = corpus_bleu(reference, translated)  
    #sacrebleu_corpus = sacrebleu.corpus_bleu( translated, reference)
    gleu_corpus = corpus_gleu(reference, translated) 

    # evaluator obj for rouge-l metric
    evaluator = Rouge(metrics=['rouge-l'],
                           limit_length=True,
                           length_limit=100,
                           length_limit_type='words',
                           apply_avg=True,
                           apply_best=False,
                           alpha=0.5, # Default F1_score
                           weight_factor=1.2,
                           stemming=True)

    
    #for result_pair in compare_results:
    for result_pair in comp_res:
        # ------------ WER        
        #res_back = wer( result_pair[0].split(' '), result_pair[1].split(' '))
        res_back = wer( result_pair[0].split(' '), result_pair[1].split(' '))

        res_wer.append( res_back )
        
        # ----------- BLEU
        indi1_gr = sentence_bleu( [result_pair[0].split(' ')], result_pair[1].split(' '), weights=(1, 0, 0, 0))  # individual 1-gram
        indi2_gr = sentence_bleu( [result_pair[0].split(' ')], result_pair[1].split(' '), weights=(0, 1, 0, 0))  # individual 2-gram
        indi3_gr = sentence_bleu( [result_pair[0].split(' ')], result_pair[1].split(' '), weights=(0, 0, 1, 0))  # individual 3-gram
        indi4_gr = sentence_bleu( [result_pair[0].split(' ')], result_pair[1].split(' '), weights=(0, 0, 0, 1))  # individual 4-gram
    
        # cumulative 2-gram, 3-gram, 4-gram bleu
        cum2_gr = sentence_bleu( [result_pair[0].split(' ')], result_pair[1].split(' '), weights=(0.5, 0.5, 0, 0))
        cum3_gr = sentence_bleu( [result_pair[0].split(' ')], result_pair[1].split(' '), weights=(0.33, 0.33, 0.33, 0))
        cum4_gr = sentence_bleu( [result_pair[0].split(' ')], result_pair[1].split(' '), weights=(0.25, 0.25, 0.25, 0.25))
        
        gleu_s = sentence_gleu( [result_pair[0].split(' ')], result_pair[1].split(' '))
        meteor = round( single_meteor_score( result_pair[0], result_pair[1]), 4)
        rouge_all = evaluator.get_scores( result_pair[1], result_pair[0])
        rouge_l_f1 = rouge_all['rouge-l']['f']
        
        bleu_indi1.append(indi1_gr)
        bleu_indi2.append(indi2_gr)
        bleu_indi3.append(indi3_gr)
        bleu_indi4.append(indi4_gr)
        bleu_cum2.append(cum2_gr)
        bleu_cum3.append(cum3_gr)
        bleu_cum4.append(cum4_gr)
        gleu_sent.append(gleu_s)
        meteor_score.append(meteor)
        rouge_score.append(rouge_l_f1)
        
    wer_mean = np.mean( res_wer )
    wer_var = np.var( res_wer )
    bleu_indi1_mean = np.mean(bleu_indi1)
    bleu_indi2_mean = np.mean(bleu_indi2)
    bleu_indi3_mean = np.mean(bleu_indi3)
    bleu_indi4_mean = np.mean(bleu_indi4)
    bleu_cum2_mean = np.mean(bleu_cum2)
    bleu_cum3_mean = np.mean(bleu_cum3)
    bleu_cum4_mean = np.mean(bleu_cum4)
    gleu_s_mean = np.mean(gleu_sent)
    meteor_s_mean = np.mean(meteor_score)
    rouge_s_mean = np.mean(rouge_score)
    
    bleus = ( bleu_indi1_mean, bleu_indi2_mean, bleu_indi3_mean, bleu_indi4_mean, bleu_cum2_mean, bleu_cum3_mean, bleu_cum4_mean, bleu_corpus)
    gleus = ( gleu_s_mean, gleu_corpus)
    
    return wer_mean, wer_var, bleus, gleus, meteor_s_mean, rouge_s_mean
    
    