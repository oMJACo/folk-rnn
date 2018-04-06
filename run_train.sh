#!/bin/bash

source activate pytorch35

EPOCHS=40
LR=17
LR_DECAY=0.80
LAYERS=3
RNN=LSTM
DROPOUT=0.2
HIDDEN="${1:-512}"
EMB=512
MODEL_OUTPUT=models/lstm_${LAYERS}_layers_lr_${LR}_decay_95_rnntype_${RNN}_dropout_${DROPOUT}_hidden_${HIDDEN}_emb_${EMB}


python main.py \
	--cuda \
	--model ${RNN} \
	--save ${MODEL_OUTPUT} \
	--epochs ${EPOCHS}  \
	--lr ${LR} \
	--lr_decay ${LR_DECAY} \
	--nlayers ${LAYERS} \
	--dropout ${DROPOUT} \
	--nhid ${HIDDEN} \
	--emsize ${EMB}
