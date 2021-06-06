import torch
import torch.nn as nn
from torch.autograd import Variable
import argparse
import os
import json
import numpy as np
from tqdm import tqdm
import string
import matplotlib.pyplot as plt

from helpers import *
from model import *
from generate import *

def random_dataset(args,file,file_len):
    inp = torch.LongTensor(args.batch_size, args.chunk_len) 
    target = torch.LongTensor(args.batch_size, args.chunk_len)
    for bi in range(args.batch_size):
        start_index = random.randint(0, file_len - args.chunk_len) # take usable starting index, not exceeding path size
        end_index = start_index + args.chunk_len + 1 

        chunk = file[start_index:end_index]
        inp[bi] = char_tensor(chunk[:-1]) # from first until second to last
        target[bi] = char_tensor(chunk[1:]) # from second until last
    inp = Variable(inp) # get correct variable type
    target = Variable(target) # get correct variable type
    if args.cuda: # move to device for faster execution
        inp = inp.cuda() 
        target = target.cuda()
    return inp, target


def consequent_dataset(args, num_batches, file, file_len):
    inp = torch.LongTensor(args.batch_size, args.chunk_len)
    target = torch.LongTensor(args.batch_size, args.chunk_len)
    end_index = args.chunk_len*num_batches*args.batch_size + (args.batch_size*num_batches) 
    end_reached = False
    for bi in range(args.batch_size):
        start_index = end_index

        if (end_reached == True):
            start_index = random.randint(0, file_len - args.chunk_len - 1)

        if (start_index + args.chunk_len + 1 > file_len):  # if we ended after the last char of the file, come back to get a correct chunk len
            start_index = file_len - args.chunk_len - 1
            end_reached = True

        end_index = start_index + args.chunk_len + 1 # Adding 1 to create target
        chunk = file[start_index:end_index]

        inp[bi] = char_tensor(chunk[:-1])
        target[bi] = char_tensor(chunk[1:])
    inp = Variable(inp)
    target = Variable(target)
    if args.cuda:
        inp = inp.cuda()
        target = target.cuda()
    return inp, target

def save(args):
    save_filename = 'Save/'
    if modelName is not None:
        save_filename += os.path.splitext(os.path.basename(args.train))[0] +'_'+modelName+ '.pt'
    else:
        save_filename += os.path.splitext(os.path.basename(args.train))[0] + '.pt'

    jsonName = save_filename + '.json'
    with open(jsonName, 'w') as json_file:
        json.dump(vars(args), json_file)
    saveLossesName = save_filename+'.csv'
    if(args.valid is not None):
        np.savetxt(saveLossesName, np.column_stack((train_losses, valid_losses)), delimiter=",", fmt='%s', header='Train,Valid')
    else:
        np.savetxt(saveLossesName, train_losses, delimiter=",", fmt='%s', header='Train')
    torch.save(decoder, save_filename)
    print('Saved as %s' % save_filename)

def savemodel(args):
    save_filename = 'Save/'
    directoryCheckpoint = 'Save/'+modelName
    if not os.path.exists(directoryCheckpoint):
        os.makedirs(directoryCheckpoint)
    if modelName is not None:
        directoryCheckpoint +='/'+ os.path.splitext(os.path.basename(args.train))[0] +'_'+modelName+ '_Checkpoint' +'.pt'
    else:
        directoryCheckpoint +='/'+ os.path.splitext(os.path.basename(args.train))[0] + '_Checkpoint'+'.pt'

    torch.save(decoder, directoryCheckpoint)

# produce train and valid given a training file and a split percentage
def split_data(fileTrain, split):
    print(f"Dividing train data: {100-split}% train - {split}% validation")
    file_lenTrain = len(fileTrain)
    start_index = random.randint(0, file_lenTrain - file_lenTrain//split) # take random start for validation
    end_index = start_index + file_lenTrain//split # accordingly set the end index
    fileValid = fileTrain[start_index:end_index] 
    tmp_fileTrain = ""
    if start_index != 0:
        tmp_fileTrain += fileTrain[:start_index]
    tmp_fileTrain += fileTrain[end_index:]
    fileTrain = tmp_fileTrain
    file_lenTrain, file_lenValid = len(fileTrain), len(fileValid)
    return fileTrain, file_lenTrain, fileValid, file_lenValid



# Initialize models and start training

if __name__ == '__main__':

    # Parse command line arguments
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--train', type=str)                                # string for input train file
    argparser.add_argument('--valid', type=str)                                # string for input valid file
    argparser.add_argument('--model', type=str, default="lstm")                # string for deciding model architecture (gru - lstm - bilstm)
    argparser.add_argument('--n_epochs', type=int, default=100)                # number of epochs
    argparser.add_argument('--print_every', type=int, default=2)               # number of epochs after which generate string
    argparser.add_argument('--hidden_size', type=int, default=512)             # number of nodes in each layer
    argparser.add_argument('--embedding_size', type=int, default=200)          # number of embedding dimensions
    argparser.add_argument('--n_layers', type=int, default=4)                  # number of layers in architecture
    argparser.add_argument('--dropout', type=float, default=0.5)               # in-layer dropout
    argparser.add_argument('--learning_rate', type=float, default=0.01)        # learning rate
    argparser.add_argument('--chunk_len', type=int, default=200)               # lenght of sequence of text
    argparser.add_argument('--batch_size', type=int, default=100)              # batch size for efficient computation
    argparser.add_argument('--batch_type', type=int, default=0)                # 0 - random training selection | 1 - sequential training selection
    argparser.add_argument('--max_batches', type=int, default=None)            # limit number of max batches
    argparser.add_argument('--early_stopping', type=int, default=50)           # if validation is not updated enough times, stop early
    argparser.add_argument('--optimizer', type=str, default="adam")            # choose optimizer (adam | rmsprop)
    argparser.add_argument('--cuda', action='store_true')                      # use or not gpu
    argparser.add_argument('--modelname', type=str, default=None)              # personalize file ending name
    argparser.add_argument('-t', '--temperature', type=float, default=1.0)     # decide if model decisions are more or less "sure"
    argparser.add_argument('--split', type=int, default=0)                     # choose how big to create validation partition
    argparser.add_argument('--weight_decay', type=float, default=None)         # different regularizing aspect
    args = argparser.parse_args()
    if args.cuda:
        print("Using CUDA")

    # get training data
    fileTrain, file_lenTrain = read_file(args.train)

    if args.split != 0: # split train into 90-10 train-valid
        fileTrain, file_lenTrain, fileValid, file_lenValid = split_data(fileTrain, args.split)
        args.valid = 0 # so to trigger train+valid execution

    numFileBatches = math.ceil(file_lenTrain/((args.batch_size*args.chunk_len)+args.batch_size)) # to understand how many iter per epoch
    if args.max_batches is not None:
        numFileBatches = min(numFileBatches, args.max_batches) # limit computation
    try:
        if args.split == 0: #try to see if valid file was given and split not already done
            fileValid, file_lenValid = read_file(args.valid) 
        numValidBatches = math.ceil(file_lenValid/((args.batch_size*args.chunk_len)+args.batch_size))
        early_stopping_patience = args.early_stopping
    except:
        print('No validation data supplied')
    if(args.modelname is None):
        print('No model name supplied -> Model checkpoint disabled')
    modelName = args.modelname

    all_characters = string.printable
    n_characters = len(all_characters)

    decoder = CharRNN(
        args.batch_size,                      # input size, since every input is a scalar
        args.hidden_size,                     # given by arg
        n_characters,                         # output size, since distrib over all possible chars
        args.embedding_size,                  # internal space mapping
        model=args.model,
        n_layers=args.n_layers,
        dropout=args.dropout,
        learning_rate=args.learning_rate,
        chunk_len= args.chunk_len,
        batch_size=args.batch_size,
        gpu = args.cuda,
        optimizer = args.optimizer,
        temperature = args.temperature,
        weight_decay = args.weight_decay
    )

    batch_type = args.batch_type # all batches sequentially or same amount of batch with random pick

    start = time.time()
    train_losses = []
    train_acc = []
    train_perp = []
    valid_losses = []
    valid_accs = []
    valid_perp = [] 
    valid_loss_best = np.inf
    valid_acc_best = -np.inf
    patience = 1 # for early stop

    print("Number of batches: ",numFileBatches)
    try:
        print("Training for %d epochs..." % args.n_epochs)

        for epoch in tqdm(range(1, args.n_epochs + 1)):
            numBatches = 0
            numBatchesValid = 0
            loss_avg = 0
            acc_avg = 0
            while(numBatches < numFileBatches) :
                if(batch_type == 0): ### Sampling batches at random
                    loss, acc = decoder.train(*random_dataset(args,fileTrain,file_lenTrain),validation=False, verbose=epoch>args.n_epochs/2)
                elif(batch_type == 1): ### Get consequent batches of chars without replacement
                    loss, acc = decoder.train(*consequent_dataset(args, numBatches,fileTrain, file_lenTrain),validation=False,  verbose=numBatches==numFileBatches-1)
                loss_avg += loss
                acc_avg += acc
                numBatches += 1
            loss_avg /= numFileBatches
            acc_avg /= numFileBatches
            # collect data
            train_losses.append(loss_avg)
            train_acc.append(acc_avg)
            train_perp.append(np.exp(loss_avg))
            if args.valid is not None:
                valid_loss_avg = 0
                valid_acc_avg = 0
                while(numBatchesValid < numValidBatches) :
                    valid_loss, valid_acc = decoder.train(*consequent_dataset(args,numBatchesValid,fileValid,file_lenValid),verbose=False, validation=True)
                    valid_loss_avg += valid_loss
                    valid_acc_avg += valid_acc
                    numBatchesValid += 1
                valid_loss_avg /= numValidBatches
                valid_acc_avg /= numValidBatches
                valid_losses.append(valid_loss_avg)
                valid_accs.append(valid_acc_avg)
                valid_perp.append(np.exp(valid_loss_avg))
                if valid_loss_avg < valid_loss_best: # print best model values and save it
                    if(args.modelname is not None):
                        print("New best checkpoint: (loss: %.4f, acc: %4f), old: (loss: %.4f, acc: %4f)" % (valid_loss_avg,valid_acc_avg,valid_loss_best,valid_acc_best))
                        savemodel(args)
                    valid_loss_best = valid_loss_avg
                    valid_acc_best = valid_acc_avg
                    args.early_stopping = epoch
                    patience = 1 # restart early stopping
                else:
                    patience += 1
                    if(patience >= early_stopping_patience):
                        break

            if epoch % args.print_every == 0: # generate data
                if args.valid is not None:
                    print('[%s (%d %d%%) Train: %.4f Valid: %.4f]' % (time_since(start), epoch, epoch / args.n_epochs * 100, loss_avg, valid_loss_avg))
                else:   
                    print('[%s (%d %d%%) Train: %.4f]' % (time_since(start), epoch, epoch / args.n_epochs * 100, loss_avg))
                print(generate(decoder, 'I think that wom', 200, cuda=args.cuda, model=args.model), '\n')
            print("\nLOSS: ",loss_avg, " - ACCURACY: ", acc_avg, " - PERPLEXITY: ", np.exp(loss_avg))

        print("Saving...")
        save(args) # save model

        # plot metrics
        fig, axs = plt.subplots(3,1)
        axs[0].plot(train_losses, color='blue', label='train')
        if args.valid is not None: axs[0].plot(valid_losses, color='red', label='valid')
        axs[1].plot(train_acc, color='blue', label='train')
        if args.valid is not None: axs[1].plot(valid_accs, color='red', label='valid')
        axs[2].plot(train_perp, color='blue', label='train')
        if args.valid is not None: axs[2].plot(valid_perp, color='red', label='valid')
        plt.legend()
        axs[0].title.set_text("LOSS")
        axs[1].title.set_text("ACCURACY")
        axs[2].title.set_text("PERPLEXITY")
        fig.tight_layout()
        plt.show()

    except KeyboardInterrupt:
        print("Saving before quit...")
        save(args)

