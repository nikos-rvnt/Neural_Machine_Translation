
# -*- coding: utf-8 -*-
"""
 
    Encoder & Attention Decoeder classes
 
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



###############################################################################
#
#                              GRU Encoder
#
###############################################################################
class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_lay = 2, dropout = 0.1):
        super( Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.n_layers = num_lay
        self.embedding = nn.Embedding(input_size, 512)
        self.dropout = dropout
        #self.embedding = nn.Embedding(input_size, embedding_dim=512)
        #print(embedded)
        self.gru = nn.GRU( 512, hidden_size, num_layers = self.n_layers, dropout=self.dropout)
        

    def forward(self, input, batch_size, hidden):
        embedded = self.embedding(input).view(1, batch_size, -1)
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self, batch_size):
        return torch.zeros( self.n_layers, batch_size, self.hidden_size, device=device)


    
###############################################################################
#
#                    Luong/Bahdanau Attention Layer
#        
###############################################################################
class Attention(torch.nn.Module):
    def __init__(self, method, hidden_size):
        
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.method = method
        if self.method not in ['dot', 'general', 'concat', 'bahdanau']:
            raise ValueError( self.method, " is not a valid attention score name.")
        
        if self.method == 'general':
            self.atten = torch.nn.Linear(self.hidden_size, hidden_size)
        elif self.method == 'concat':
            self.atten = torch.nn.Linear(self.hidden_size * 2, hidden_size)
            self.v = torch.nn.Parameter(torch.FloatTensor(hidden_size))
        elif self.method == 'bahdanau': 
            self.attenW = torch.nn.Linear( self.hidden_size, hidden_size)
            self.attenU = torch.nn.Linear( self.hidden_size, hidden_size)
            self.v = torch.nn.Parameter(torch.FloatTensor(hidden_size))

    # dot score: (ht)^T*hs
    def dot_func(self, hidden, encoder_output): 
        return torch.sum( hidden * encoder_output, dim=2 )

    # general score: (ht)^T*W*hs
    def general_func(self, hidden, encoder_output):
        energy = self.atten(encoder_output)
        return torch.sum( hidden * energy, dim=2 )

    # concat score: v*tanh( [(ht)^T;hs]*W
    def concat_func(self, hidden, encoder_output):
        energy = self.atten(torch.cat((hidden.expand(encoder_output.size(0), -1, -1), encoder_output), 2)).tanh()
        return torch.sum( self.v * energy, dim=2 )

    # bahdanau score: v*tanh( W(ht)^T + Uhs)
    def bahdanau_func(self, hidden, encoder_output): 
        energy1 = self.attenW( hidden.expand(encoder_output.size(0), -1, -1) )
        energy2 = self.attenU( encoder_output )
        energy = (energy1 + energy2).tanh()
        return torch.sum( self.v * energy, dim=2 )

    def forward(self, hidden, encoder_outputs):

        # for the chosen attention score func --> attention wights (energy)        
        if self.method == 'dot':
            attn_energies = self.dot_func(hidden, encoder_outputs)
        elif self.method == 'general':
            attn_energies = self.general_func(hidden, encoder_outputs)
        elif self.method == 'concat':
            attn_energies = self.concat_func(hidden, encoder_outputs)
        elif self.method == 'bahdanau':
            attn_energies = self.bahdanau_func(hidden, encoder_outputs)

        # transpose max_length, batch_size 
        attn_energies = attn_energies.t()

        # normalized probability scores 
        return F.softmax(attn_energies, dim=1).unsqueeze(1)

###############################################################################
#
#                             Attention GRU Decoder
#
###############################################################################
class Attention_Decoder(nn.Module):
    def __init__(self, attn_model, hidden_size, output_size, n_layers=1, dropout=0.1):
        super( Attention_Decoder, self).__init__()

        # Keep for reference
        self.attention_model = attn_model
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout = dropout

        # Define layers
        self.embedding = nn.Embedding(self.output_size, embedding_dim=512)
        #self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.embedding_dropout = nn.Dropout(dropout)
        self.gru = nn.GRU( 512, hidden_size, n_layers, dropout=(0 if n_layers == 1 else dropout))
        self.concat = nn.Linear(hidden_size * 2, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

        self.atten = Attention(attn_model, hidden_size)

    def forward(self, input_step, batch_size, last_hidden, encoder_outputs):
        
        # we run this one step (word) at a time
        # get embedding of current input word
        embedded = self.embedding(input_step).view(1, batch_size, -1)
        embedded = self.embedding_dropout(embedded)
        
        # forward through unidirectional GRU
        rnn_output, hidden = self.gru(embedded, last_hidden)
        
        # attention weights from the current GRU output
        attn_weights = self.atten(rnn_output, encoder_outputs)
        
        # attention weights * encoder outputs = weighted sum context vector
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))
        
        # concat weighted context vector, GRU output using Luong eq. 5
        rnn_output = rnn_output.squeeze(0)
        context = context.squeeze(1)
        concat_input = torch.cat((rnn_output, context), 1)
        concat_output = torch.tanh(self.concat(concat_input))
        
        # predict next word using Luong eq. 6
        output = self.out(concat_output)
        output = F.log_softmax(output, dim=1)
        
        return output, hidden
    

