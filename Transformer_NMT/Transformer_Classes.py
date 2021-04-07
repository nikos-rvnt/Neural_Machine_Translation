# -*- coding: utf-8 -*-
"""
 
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.nn import Transformer
from torch.nn import TransformerEncoderLayer, TransformerDecoderLayer, TransformerEncoder, TransformerDecoder
from torch.nn import LayerNorm
from torch.nn.init import xavier_uniform_


class Transformer_Model(nn.Module):

    def __init__(self, ntoken_in, ntoken_out, embed_size, drop_it, trns_model='base'):
        super(Transformer_Model, self).__init__()
 
        self.src_mask = None
        self.trg_mask = None 
        self.drop = drop_it
        self.PAD_token = 2
        self.src_pad_mask = None
        self.trg_pad_mask = None
        
        self.embed_in = nn.Embedding( ntoken_in, embed_size)
        self.embed_out = nn.Embedding( ntoken_out, embed_size)
        self.pos_encoder = PositionalEncoding( embed_size, self.drop, max_len=5000)
        
        
        if trns_model=='base':
            # base model
            encoder_layer = TransformerEncoderLayer(512, 8, 2048, 0.1)
            encoder_norm = LayerNorm(512)
            self.encoder = TransformerEncoder(encoder_layer, 6, encoder_norm)
            
            decoder_layer = TransformerDecoderLayer(512, 8, 2048, 0.1)
            decoder_norm = LayerNorm(512)
            self.decoder = TransformerDecoder(decoder_layer, 6, decoder_norm)
            
        else:
            # big model
            encoder_layer = TransformerEncoderLayer(1024, 16, 4096, 0.3)
            encoder_norm = LayerNorm(1024)
            self.encoder = TransformerEncoder(encoder_layer, 6, encoder_norm)
 
            decoder_layer = TransformerDecoderLayer(1024, 16, 4096, 0.3)
            decoder_norm = LayerNorm(1024)
            self.decoder = TransformerDecoder(decoder_layer, 6, decoder_norm)
            
        self.ninp = embed_size
        self.linear_dec = nn.Linear( embed_size, ntoken_out)
        
        # initialise embedding & linear layer parameters
        self.init_weights()
        # initialise transformer parameters
        self.reset_params()

        
    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        self.embed_in.weight.data.uniform_( -initrange, initrange)
        self.embed_out.weight.data.uniform_( -initrange, initrange)
        self.linear_dec.bias.data.zero_()
        self.linear_dec.weight.data.uniform_( -initrange, initrange)
        
    def reset_params(self):
        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)


    def forward(self, src, trg):
      
#        if self.src_mask is None or self.src_mask.size(0) != len(src):
#            device = src.device
#            s_mask = self._generate_square_subsequent_mask(len(src)).to(device)
#            self.src_mask = s_mask
        
        # square attention mask is required because the self-attention layers in
        # TransformerDecoder are only allowed to attend the earlier positions in the sequence
        if self.trg_mask is None or self.trg_mask.size(0) != len(trg):
            device = trg.device
            t_mask = self._generate_square_subsequent_mask(len(trg)).to(device)
            self.trg_mask = t_mask
 
        # create key pad mask per batch 
        src_pad_mask = (src != self.PAD_token).type(torch.ByteTensor)
        trg_pad_mask = (trg != self.PAD_token).type(torch.ByteTensor)
        
        src.cpu()
        trg.cpu()
        # input through word embeddings
        src = self.embed_in(src) * math.sqrt(self.ninp)
        trg = self.embed_out(trg) * math.sqrt(self.ninp)

        # add positional encoding to input/output
        src = self.pos_encoder(src)
        trg = self.pos_encoder(trg)

        # encoder-decoder instead of standalone transformer module
#        memory = self.encoder(src, mask=self.src_mask, src_key_padding_mask=self.src_pad_mask)
#        output = self.decoder(trg, memory, tgt_mask=self.trg_mask, tgt_key_padding_mask=self.trg_pad_mask)
        memory = self.encoder(src, mask=self.src_mask)
        output = self.decoder(trg, memory, tgt_mask=self.trg_mask)
        
        # output through a linear layer and then softmax
        output = self.linear_dec(output)
        output = F.log_softmax( output, dim=-1)

        return output
    
    def forward_eval_enc(self, src):
        
        src = self.embed_in(src) * math.sqrt(self.ninp)
        src = self.pos_encoder(src)

        mem = self.encoder( src, mask=None, src_key_padding_mask=None)
        
        return mem
    
    def forward_eval_dec(self, trg, memo):
        
        if self.trg_mask is None or self.trg_mask.size(0) != len(trg):
            device = trg.device
            t_mask = self._generate_square_subsequent_mask(len(trg)).to(device)
            self.trg_mask = t_mask

        trg = self.embed_out(trg) * math.sqrt(self.ninp)
        # add positional encoding to input/output
        trg = self.pos_encoder(trg)
        
        # transformer
        output = self.decoder( trg, memo, tgt_mask=self.trg_mask,
                memory_mask=None, tgt_key_padding_mask=None,
                memory_key_padding_mask=None)        
        # output through a linear layer and then softmax
        output = self.linear_dec(output)
        out = F.log_softmax( output, dim=-1)
        
        return out
    
    
class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
    
    
    
    