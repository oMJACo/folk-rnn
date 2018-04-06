#!/bin/bash

source activate pytorch35

CHECKPOINT=models/lstm_3_layers_lr_18_decay_95_rnntype_LSTM_dropout_0.2_hidden_512_emb_512/model.pt
WORDS=145
OUTFILE='generated.txt'

python generate.py \
	--checkpoint ${CHECKPOINT} \
	--words ${WORDS} \
	--outf ${OUTFILE}

