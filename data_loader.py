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
import ipdb

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

def token_to_ix_src(batch):
    token_to_ix = {}
    ipdb.set_trace()
    for src, trg in zip(batch.src, batch.trg):
        for token in src:
            if token not in token_to_ix:
                token_to_ix[token] = len(token_to_ix)
    return token_to_ix

def token_to_ix_trg(batch):
    token_to_ix = {}
    for src, trg in zip(batch.src, batch.trg):
        for token in trg:
            if token not in token_to_ix:
                token_to_ix[token] = len(token_to_ix)
    return token_to_ix

def prepare_sequence(tune, to_ix):
    idxs = [to_ix[token] for token in tune]
    tensor = torch.LongTensor(idxs)
    return autograd.Variable(tensor)

EMBEDDING_DIM = 100
HIDDEN_DIM = 100

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
        lstm_out, self.hidden = self.lstm(
                embeds.view(len(tune), 1, -1), self.hidden)
        trg_space = self.hidden2trg(lstm_out.view(len(tune), -1))
        trg_scores = F.log_softmax(trg_space, dim=1)

#Train Model

model = LSTM(EMBEDDING_DIM, HIDDEN_DIM, len(TEXT.vocab), len(TGT.vocab))
loss_function = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr = 0.1)

for tune in train_iter:
    for src, trg in zip(tune.src, tune.trg):
        ipdb.set_trace()
        model.zero_grad()
        model.hidden = model.init_hidden()

        #tune_in = prepare_sequence(src, token_to_ix_src(tune))
        #targets = prepare_sequence(trg, token_to_ix_trg(tune))

        target_scores = model(src)
        print(target_scores)
        ipdb.set_trace()

        loss = loss_function(target_scores, trg)
        loss.backward()
        optimizer.step()

print(tag_scores)






