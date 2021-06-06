import torch
import torch.nn as nn
from torch.autograd import Variable

class CharRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, embedding_size, model="gru", n_layers=1,
     dropout = 0, gpu = True, batch_size = 32, chunk_len = 30, learning_rate = 0.001, optimizer = "adam", temperature=1.0, weight_decay = None):
        super(CharRNN, self).__init__()
        self.model = model.lower()           # normalize string
        self.input_size = input_size         # take various sizes..
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.gpu = gpu
        self.batch_size = batch_size
        self.chunk_len = chunk_len
        self.optimizer = optimizer
        self.temperature = temperature

        self.encoder = nn.Embedding(self.input_size, self.embedding_size)
        # define different behaviours given different models
        if self.model == "gru":
            self.rnn = nn.GRU(self.embedding_size, self.hidden_size, self.n_layers, dropout=dropout)
        elif self.model == "lstm":
            self.rnn = nn.LSTM(self.embedding_size, self.hidden_size, self.n_layers, dropout=dropout)
        elif self.model == "bilstm":
            # BI-LSTM 
            self.lstm_forward = nn.LSTM(self.embedding_size, self.hidden_size)
            self.lstm_backward = nn.LSTM(self.embedding_size, self.hidden_size)
            # final LSTM
            self.final_lstm = nn.LSTM(2*self.hidden_size, 2*self.hidden_size)
        self.decoder = nn.Linear(self.hidden_size, self.output_size) if self.model != "bilstm" else nn.Linear(2*self.hidden_size, self.output_size)
        if self.optimizer == "adam":
            if weight_decay is not None: # regularize if needed
                self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate, weight_decay=weight_decay)
            else:
                self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss()
        if self.gpu:
            self.cuda()

    def forward(self, input, backward_input, hidden, batch_size = None):
        batch_size = batch_size if batch_size is not None else self.batch_size # to differentiate between train/valid and generation's custom batchsize
        encoded = self.encoder(input)
        if self.model == 'bilstm':
            encoded_backward = self.encoder(backward_input) 
            hidden0 = hidden[0] # take hidden for forward lstm
            hidden1 = hidden[1] # take hidden for backward lstm
            hidden2 = hidden[2] # take hidden for final lstm
            output1, hidden0 = self.lstm_forward(encoded.view(1, batch_size, -1), hidden0) 
            output2, hidden1 = self.lstm_backward(encoded_backward.view(1, batch_size, -1), hidden1)
            input_tensor = torch.cat((output1, output2), 2) # concat the results on last dimension
            output, hidden2 = self.final_lstm(input_tensor, hidden2)
            hidden = (hidden0, hidden1, hidden2) # reconstruct to go back to unitary computation below
        else:
            output, hidden = self.rnn(encoded.view(1, batch_size, -1), hidden)
        output = self.decoder(output.view(batch_size, -1))
        output = output.div(self.temperature) # temperature even in training, sharper results!
        return output, hidden

    def init_hidden(self, batch_size):
        if self.model == "lstm": # remember + hidden
             return (Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size)),
                    Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size)))
        elif self.model == "bilstm": # 3 times (remember + hidden)
            hidden_fwd = (Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size)), Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size)))
            hidden_bwd = (Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size)), Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size)))
            hidden_final = (Variable(torch.zeros(self.n_layers, batch_size, 2*self.hidden_size)), Variable(torch.zeros(self.n_layers, batch_size, 2*self.hidden_size)))
            return (hidden_fwd, hidden_bwd, hidden_final)
        return Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size)) # state is composed only of a tensor

    
    def train(self,inp, target, validation, verbose):
        self.zero_grad() # start of batch
        loss,acc = 0, 0
        hidden = self.init_hidden(self.batch_size)
        if self.cuda: # allocate on gpu if needed
            if self.model == "gru":
                hidden = hidden.cuda()
            elif self.model == "lstm":
                hidden = (hidden[0].cuda(), hidden[1].cuda())
            else:
                hidden = ((hidden[0][0].cuda(), hidden[0][1].cuda()), (hidden[1][0].cuda(), hidden[1][1].cuda()), (hidden[2][0].cuda(), hidden[2][1].cuda()))
        for c in range(self.chunk_len): # for every char in the sentence length
            output, hidden = self(inp[:, c], inp[:, self.chunk_len - 1 - c], hidden) # feed forward
            # prepare for loss + append values for plotting
            pred, actual = output.view(self.batch_size, -1),target[:, c]
            _,predicted = torch.max(pred.data, 1) # to avoid TEACHER FORCING --> give "predicted" as input
            acc += (predicted == actual).sum().item()
            loss += self.criterion(pred, actual)       
         ### The losses are averaged across observations for each minibatch (see doc CrossEntropyLoss)
        if not validation:
            loss.backward() # backprop
            self.optimizer.step() 
        currentAcc = acc / (self.chunk_len * predicted.size(0))
        currentLoss = loss.item()/ (self.chunk_len * predicted.size(0))
        return currentLoss, currentAcc