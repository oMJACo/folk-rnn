#!/bin/bash

EPOCHS=2
LR=1
LR_DECAY=0.95

python main.py \
	--cuda \
	--epochs ${EPOCHS}  \
	--lr ${LR}
