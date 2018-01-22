#!/bin/bash

THEANO_FLAGS='floatX=float32,device=cuda,dnn.enabled=False' python test_gpu.py
