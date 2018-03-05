#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  2 11:12:10 2018

@author: murraycowan
"""

from torchtext import data
from torchtext import datasets
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd

# Approach 1:
# set up fields
TEXT = data.Field(lower=False, batch_first=True, eos_token="<eos>")
TGT  = data.Field(lower=False, batch_first=True, eos_token="<eos>")

# make splits for data
train = datasets.MusicDataset(path='/home/mcowan/Dissertation/single_line_data/',
                              exts=('train_src', 'train_tgt'), 
                              fields=[('src',TEXT),('trg',TGT)])

valid = datasets.MusicDataset(path='/home/mcowan/Dissertation/single_line_data/',
                              exts=('valid_src', 'valid_tgt'),
                              fields=[('src',TEXT),('trg', TGT)])

# build the vocabulary
TEXT.build_vocab(train)
TGT.build_vocab(train)

# print vocab information
print('len(TEXT.vocab)', len(TEXT.vocab))
print('len(TGT.vocab)', len(TGT.vocab))

# make iterator for splits
train_iter, valid_iter = data.BucketIterator.splits(
    (train, valid), batch_size=(500), device=-1)

train_batch = next(iter(train_iter))
valid_batch = next(iter(valid_iter))
#can use batch.src, batch.trg

EMBEDDING_DIM = 6
HIDDEN_DIM = 6

class LSTM(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, src_vocab_size, trg_vocab_size):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim

        self.word_embeddings = nn.Embedding(src_vocab_size, embedding_dim)

        self.lstm = nn.LSTM(embedding_dim, hidden_dim)

        self.hidden2trg = nn.Linear(hidden_dim, trg_vocab_size)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (autograd.Variable(torch.zeros(1,1,self.hidden_dim)),
                autograd.Variable(torch.zeros(1,1,self.hidden_dim)))

    def forward(self, tune):
        embeds = self.word_embeddings(tune)
        lst_out, self.hidden = self.lstm(
                embeds.view(len(tune), 1, -1), self.hidden)
        trg_space = self.hidden2trg(lstm.out.view(len(tune), -1))
        trg_scores = F.log_softmax(trg_space)

#Train Model

model = LSTM(EMBEDDING_DIM, HIDDEN_DIM, len(TEXT.vocab), len(TGT.vocab))
loss_function = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr = 0.1)

trg_scores = model()
print(tag_scores)

