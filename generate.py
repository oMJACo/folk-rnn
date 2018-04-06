###############################################################################
# Language Modeling on Penn Tree Bank
#
# This file generates new sentences sampled from the language model
#
###############################################################################

import argparse

import torch
from torch.autograd import Variable
from torchtext import data
from torchtext import datasets
import random


parser = argparse.ArgumentParser(description='PyTorch Wikitext-2 Language Model')

# Model parameters.
parser.add_argument('--data', type=str, default='./data/wikitext-2',
                    help='location of the data corpus')
parser.add_argument('--vocab', type=str, default='models/vocab.pt',
                    help='location of the data corpus')
parser.add_argument('--checkpoint', type=str, default='models/lstm_4_layers_lr_18_decay_95_rrntype_LSTM/model.pt',
                    help='model checkpoint to use')
parser.add_argument('--outf', type=str, default='generated.txt',
                    help='output file for generated text')
parser.add_argument('--words', type=int, default='145',
                    help='number of words to generate')
parser.add_argument('--seed', type=int, default=5,
                    help='random seed')
parser.add_argument('--cuda', action='store_false',
                    help='use CUDA')
parser.add_argument('--temperature', type=float, default=1.0,
                    help='temperature - higher will increase diversity')
parser.add_argument('--log-interval', type=int, default=100,
                    help='reporting interval')
args = parser.parse_args()

TEXT = data.Field(lower=False, batch_first=True, eos_token="<eos>")

train_data=datasets.MusicDataset(path='/home/mcowan/Project/folk-rnn/data/',
                                 exts=('train_src', 'train_tgt'),
                                 fields=[('src',TEXT),('trg',TEXT)])

val_data=datasets.MusicDataset(path='/home/mcowan/Project/folk-rnn/data/',
                               exts=('valid_src','valid_tgt'),
                               fields=[('src',TEXT),('trg', TEXT)])
import os.path
import pickle

vocab = None
if not os.path.isfile(args.vocab):
    print('Saving vocabulary...')
    TEXT.build_vocab(train_data)
    vocab = TEXT.vocab
    with open(args.vocab, 'wb') as f:
        pickle.dump(vocab, f)
else:
    print('Loading vocabulary...')
    vocab = pickle.load(open(args.vocab, 'rb'))

vocab_length = len(vocab)
# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        torch.cuda.manual_seed(args.seed)

if args.temperature < 1e-3:
    parser.error("--temperature has to be greater or equal 1e-3")

print('Loading model...')
with open(args.checkpoint, 'rb') as f:
    model = torch.load(f)
model.eval()

if args.cuda:
    print('running on GPU')
    model.cuda()
else:
    print('running on CPU')
    model.cpu()

ntokens = vocab_length
hidden = model.init_hidden(1)

# import ipdb
# ipdb.set_trace()

measure_idxs = [49,58,63,77,81,68]
keySig_idxs = [41,56,59,62]

#input = Variable(torch.rand(1, 1).mul(ntokens).long(), volatile=True)
input = Variable(torch.LongTensor(1,1))
#input[0] = random.choice(measure_idxs)
input.data.fill_(random.choice(measure_idxs))

import ipdb
ipdb.set_trace()

if args.cuda:
    input.data = input.data.cuda()

with open(args.outf, 'w') as outf:
    output_tune = ''
    for i in range(args.words):
        output, hidden = model(input, hidden)
        word_weights = output.squeeze().data.div(args.temperature).exp().cpu()
        word_idx = torch.multinomial(word_weights, 1)[0]
        input.data.fill_(word_idx)
        word = vocab.itos[word_idx]
        if word == '<eos>':
            break
        else:
            output_tune += word + ' '

    outf.write(output_tune + '\n')
    print(output_tune)
