#!/bin/bash

CONFIG_FILE=config5
DATASET=data/allabcwrepeats_parsed_wot

THEANO_FLAGS='floatX=float32,device=cuda,dnn.enabled=False' python train_rnn.py ${CONFIG_FILE} ${DATASET}
