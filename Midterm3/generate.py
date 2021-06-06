#!/usr/bin/env python
# https://github.com/zutotonno/char-rnn.pytorch

import torch
import os
import argparse
import string

from helpers import *
from model import *

all_characters = string.printable
n_characters = len(all_characters)

def generate(decoder, prime_str='A', predict_len=100, temperature=0.8, cuda=False, model='lstm'):
    hidden = decoder.init_hidden(1) # batchsize to expect
    prime_input = Variable(char_tensor(prime_str).unsqueeze(0)) # from string to torch tensor

    if cuda: # move variables to gpu if needed
        if isinstance(hidden, tuple):
            if model=='bilstm':
                hidden = ((hidden[0][0].cuda(), hidden[0][1].cuda()), (hidden[1][0].cuda(), hidden[1][1].cuda()), (hidden[2][0].cuda(), hidden[2][1].cuda()))
            else:
                hidden = (hidden[0].cuda(), hidden[1].cuda())
        else:
            hidden = hidden.cuda()
        prime_input = prime_input.cuda()
    predicted = prime_str 

    # Use priming string to "build up" hidden state
    for p in range(len(prime_str) - 1):
        _, hidden = decoder(prime_input[:, p], prime_input[:, len(prime_str) - 1 - p], hidden, 1)
        
    inp = prime_input[:, -1] # take next input to model
    
    for p in range(predict_len):
        output, hidden = decoder(inp, inp, hidden, 1) 

        # Sample from the network as a multinomial distribution
        output_dist = output.data.view(-1).div(temperature).exp()
        top_i = torch.multinomial(output_dist, 1)[0] # sample

        # Add predicted character to string and use as next input
        predicted_char = all_characters[top_i]
        
        predicted += predicted_char
        inp = Variable(char_tensor(predicted_char).unsqueeze(0))
        if cuda:
            inp = inp.cuda()
    return predicted

# Run as standalone script
if __name__ == '__main__':

# Parse command line arguments
    argparser = argparse.ArgumentParser()
    argparser.add_argument('filename', type=str)                              # model file
    argparser.add_argument('-p', '--prime_str', type=str, default='A')        # string which starts the text
    argparser.add_argument('--model', type=str, default='lstm')               # to understand how to initialize hidden structure
    argparser.add_argument('-l', '--predict_len', type=int, default=100)      # how much to go on
    argparser.add_argument('-t', '--temperature', type=float, default=0.8)    # temperature to influence probabilities in output
    argparser.add_argument('--cuda', action='store_true')                     # cpu or gpu
    args = argparser.parse_args()

    decoder = torch.load(args.filename) # load the model
    del args.filename
    print(generate(decoder, **vars(args))) # generate text

