#!/usr/bin/env python
#encoding: utf-8

""" Sample script of recurrent neural network language model for generating text

This code is ported from following implementation.
https://github.com/longjie/chainer-char-rnn/blob/master/sample.py

"""
import argparse
import codecs

from chainer import cuda, Variable
import cPickle as pickle

import chainer.functions as F
import chainer.links as L
from chainer import serializers

import net
import numpy as np

import sys

parser = argparse.ArgumentParser()
parser.add_argument('--model', '-m',type=str,   required=True,
                    help='model data, saved by train_ptb.py')
parser.add_argument('--vocabulary','-v',type=str,   required=True,
                    help='vocabulary data, saved by train_ptb.py')
parser.add_argument('--primetext', '-p', type=str,   default='', 
                    help='base text data, used for text generation')
parser.add_argument('--seed', '-s', type=int,   default=123, 
                    help='random seeds for text generation')
parser.add_argument('--unit', '-u',  type=int,   default=650, 
                    help='number of units')
parser.add_argument('--sample',     type=int,   default=1, 
                    help='negative value indicates NOT use random choice')
parser.add_argument('--length',     type=int,   default=20, 
                    help='length of the generated text')
parser.add_argument('--gpu',        type=int,   default=-1, 
                    help='GPU ID (negative value indicates CPU)')
args = parser.parse_args()

np.random.seed(args.seed)

# load vocabulary
vocab = pickle.load(open(args.vocabulary, 'rb'))
ivocab = {}
for c, i in vocab.items():
    ivocab[i] = c

# shoud be same as n_units , described in train_ptb.py
n_units = args.unit

lm = net.RNNLM(len(vocab), n_units, train=False)
model = L.Classifier(lm)

serializers.load_npz(args.model, model)

if args.gpu >= 0:
    cuda.init()
    model.to_gpu()

model.predictor.reset_state()

sys.stdout = codecs.getwriter('utf_8')(sys.stdout)

global prev_word
primetext = unicode(args.primetext, 'utf-8')
    
if len(primetext) > 0:
    prev_word = Variable(np.ones((1,)).astype(np.int32) * vocab[primetext])
    if args.gpu >= 0:
        prev_word = cuda.to_gpu(prev_word)

    prob = F.softmax(model.predictor(prev_word))     
    sys.stdout.write(primetext + " ") 

for i in xrange(args.length):
    prob = F.softmax(model.predictor(prev_word))
    
    if args.sample > 0:
        probability = cuda.to_cpu(prob.data)[0].astype(np.float64)
        probability /= np.sum(probability)
        index = np.random.choice(range(len(probability)), p=probability)
    else:
        index = np.argmax(cuda.to_cpu(prob.data))

    if ivocab[index] == "<eos>":
        sys.stdout.write(".")
    else:
        sys.stdout.write(ivocab[index] + " ")        

    prev_word = Variable(np.ones((1,)).astype(np.int32) * vocab[ivocab[index]])
    if args.gpu >= 0:
        prev_word = cuda.to_gpu(prev_word)
