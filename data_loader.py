#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  2 11:12:10 2018

@author: murraycowan
"""

from torchtext import data
from torchtext import datasets

# Approach 1:
# set up fields
TEXT = data.Field(lower=True, batch_first=True)

# make splits for data
train, valid = datasets.LanguageModelingDataset.splits(path='/Users/murraycowan/Desktop/Uni Stuff/Fourth Year/Dissertation/folk-rnn-master/data/allabcwrepeats_parsed', train='_train', 
                                                       validation='_valid',
                                                text_field=TEXT)

import pdb
pdb.set_trace()

# print information about the data
print('train.fields', train.fields)
print('len(train)', len(train))
print('vars(train[0])', vars(train[0])['text'][0:10])

# build the vocabulary
TEXT.build_vocab(train, vectors=GloVe(name='6B', dim=300))

# print vocab information
print('len(TEXT.vocab)', len(TEXT.vocab))

# make iterator for splits
train_iter, valid_iter = data.BucketIterator.splits(
    (train, valid), batch_size=3, bptt_len=30, device=0)

# print batch information
batch = next(iter(train_iter))
print(batch.text)
print(batch.target)

# Approach 2:
train_iter, valid_iter = datasets.WikiText2.iters(batch_size=4, bptt_len=30)

# print batch information
batch = next(iter(train_iter))
print(batch.text)
print(batch.target)