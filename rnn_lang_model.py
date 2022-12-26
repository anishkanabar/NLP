import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchtext
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import Vectors, build_vocab_from_iterator

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
seed = 53113    
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

PATH = 'gdrive/MyDrive/nlp22/hw1/datasets/'
BATCH_SIZE = 32
EMBEDDING_SIZE = 650
MAX_VOCAB_SIZE = 50000
TRAIN_DATA_SET = "lm-train.txt"
DEV_DATA_SET = "lm-dev.txt"
TEST_DATA_SET = "lm-test.txt"
BPTT_LENGTH = 32

def get_text(fn: str):
    '''
    fn: filename of text file to read
    Returns:
        list of words in the file {fn}
    '''
    with open(fn, 'r') as f:
        txt = f.read().strip().lower().split(" ")
        return txt
    
class TextStreamer(Dataset):
    def __init__(self, text: str, max_vocab_size: int, seq_len: int,
                 vocabulary = None):
        '''
        text: text to stream
        max_vocab_size: max size of vocabulary of the text
        seq_len: sequence length to stream
        '''
        self.text = text
        if not vocabulary:
            self.vocab = build_vocab_from_iterator(
                map(get_tokenizer('basic_english'), text),
                specials=['<unk>'],
                max_tokens = max_vocab_size)
            self.vocab.set_default_index(self.vocab['<unk>'])
        else:
            self.vocab = vocabulary
        self.seq_len = seq_len

    @property
    def vocab_size(self):
        # add one more for out of vocab words
        return len(self.vocab) + 1
    
    def get_vocab(self):
        return self.vocab

    def __len__(self):
        
        return (len(self.text) - 1) // self.seq_len

    def __getitem__(self, i):
        j = i * self.seq_len
        src = self.text[j: j+self.seq_len]
        tgt = self.text[j+1: j+1+self.seq_len]
        tgt_idx = self.vocab.lookup_indices(tgt)
        src_idx = self.vocab.lookup_indices(src)
        return (torch.LongTensor(src_idx), torch.LongTensor(tgt_idx))    

from torch.utils.data import Sampler
class BatchSequentialSampler(Sampler):
    r"""Samples batches, s.t. the ith elements of each batch are sequential.

    Args:
        data_source (Dataset): dataset to sample from
    """

    def __init__(self, data_source, batch_size):
        self.data_source = data_source
        self.batch_size = batch_size
        
    def __iter__(self):
        num_batches = len(self.data_source)//self.batch_size
        for i in range(num_batches):
            for j in range(self.batch_size):
                yield(j * num_batches + i)

    def __len__(self):
        return (len(self.data_source)//self.batch_size) * self.batch_size

fnames = [TRAIN_DATA_SET, DEV_DATA_SET, TEST_DATA_SET]
train_fn, val_fn, test_fn = [os.path.join(PATH, x) for x in fnames]
train_set = TextStreamer(get_text(train_fn), MAX_VOCAB_SIZE, BPTT_LENGTH)
train_sampler = BatchSequentialSampler(train_set, BATCH_SIZE)
train_loader = DataLoader(train_set, BATCH_SIZE, sampler = train_sampler,
                          drop_last = True)

val_set = TextStreamer(get_text(val_fn), MAX_VOCAB_SIZE, BPTT_LENGTH,
                      vocabulary = train_set.get_vocab())
val_sampler = BatchSequentialSampler(val_set, BATCH_SIZE)
val_loader = DataLoader(val_set, BATCH_SIZE, sampler = val_sampler,
                        drop_last = True)

test_set = TextStreamer(get_text(test_fn), MAX_VOCAB_SIZE, BPTT_LENGTH,
                      vocabulary = train_set.get_vocab())
test_sampler = BatchSequentialSampler(test_set, BATCH_SIZE)
test_loader = DataLoader(test_set, BATCH_SIZE, sampler = test_sampler,
                         drop_last = True)

src, tgt = next(iter(train_loader))
print("Constructing batches of size: {}, sequences of length: {}".format(BATCH_SIZE, BPTT_LENGTH))
print("Source batch shape: {}".format(src.shape))
print("Target batch shape: {}".format(tgt.shape))

import torch
import torch.nn as nn


class RNNLM(nn.Module):
    """ Container module with an linear encoder/embedding, an RNN module, and a linear decoder.
    """

    def __init__(self, rnn_type, vocab_size, embedding_dim, hidden_dim, num_layers, 
                 dropout=0.5):
        ''' Initialize model parameters corresponding to ---
            - embedding layer
            - recurrent neural network layer---one of LSTM, GRU, or RNN---with 
              optionally more than one layer
            - linear layer to map from hidden vector to the vocabulary
            - optionally, dropout layers.  Dropout layers can be placed after 
              the embedding layer or/and after the RNN layer. Dropout within
              an RNN is only applied when there are two or more num_layers.
            - optionally, initialize the model parameters.
            
            The arguments are:
            
            rnn_type: One of 'LSTM', 'GRU', 'RNN_TANH', 'RNN_RELU'
            vocab_size: size of vocabulary
            embedding_dim: size of an embedding vector
            hidden_dim: size of hidden/state vector in RNN
            num_layers: number of layers in RNN
            dropout: dropout probability.
            
        '''
        super(RNNLM, self).__init__()

        self.cell = True 
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.Embedding = nn.Embedding(vocab_size, embedding_dim)
        
        if rnn_type == "LSTM2":
          self.RNN = nn.LSTM(embedding_dim, hidden_dim, num_layers)
        elif rnn_type == "GRU":
          self.RNN = nn.GRU(embedding_dim, hidden_dim)
        else:
          self.RNN = nn.LSTM(embedding_dim, hidden_dim)

        self.Dropout = nn.Dropout(dropout)
        self.Linear = nn.Linear(hidden_dim, vocab_size)

    def forward(self, input, hidden0):
        ''' 
        Run forward propagation for a given minibatch of inputs using
        hidden0 as the initial hidden state.

        In LSTMs hidden0 = (h_0, c_0). 

        The output of the RNN includes the hidden vector hiddenn = (h_n, c_n).
        Return this as well so that it can be used to initialize the next
        batch.
        
        '''
        embeds = self.Embedding(input)
        output, state = self.RNN(embeds, hidden0)
        out = self.Dropout(output)
        logits = self.Linear(out)
        return logits, state

def evaluate(model, data_loader):
    '''
    Evaluate the model on the given data.
    '''

    model.eval()
    it = iter(data_loader)
    total_count = 0. # Number of target words seen
    total_loss = 0. # Loss over all target words
    with torch.no_grad():
        # No gradients need to be maintained during evaluation
        # There are no hidden tensors for the first batch, and so will default to zeros.
        hidden = None 
        for i, batch in enumerate(it):
            text, target = batch
            text, target = text.to(device), target.to(device)
            output, hidden = model(text, hidden)
            loss = loss_fn(output.view(-1, output.size(-1)), target.view(-1))
            
            total_count += np.multiply(*text.size())
            total_loss += loss.item()*np.multiply(*text.size())
                
    loss = total_loss / total_count
    model.train()
    return loss

from typing import Text
GRAD_CLIP = 1.
NUM_EPOCHS = 2

def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""
    if h is None:
        return None
    elif isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)
    
vocab_size = len(train_set.get_vocab())
model = RNNLM("LSTM", vocab_size, EMBEDDING_SIZE, EMBEDDING_SIZE, 2, 
              dropout=0.5)
model = model.to(device)

loss_fn = nn.CrossEntropyLoss() ## Used instead of NLLLoss.
learning_rate = 0.001
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
val_losses = []
best_model = None
for epoch in range(NUM_EPOCHS):
    model.train()
    it = iter(train_loader)
    # There are no hidden tensors for the first batch, and so will default to zeros.
    hidden = None
    for i, batch in enumerate(it):
        
        text, target = batch
        text, target = text.to(device), target.to(device)

        y_pred, (state_h, state_c) = model(text, hidden)
        loss = loss_fn(y_pred.transpose(1, 2), target)

        state_h = state_h.detach()
        state_c = state_c.detach()

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()

        if i % 100 == 0 and i > 0:
            print(f'At iteration {i} the training loss is {loss:.3f}.')

        if i % 10000 == 0 and i > 0:
            val_loss = model.evaluate(val_loader)
            print(f'At iteration {i} the validation loss is {val_loss:.3f}.')

'''
Evaluate the loss of best_model on the test set and compute its perplexity.
'''
test_loss = evaluate(best_model, test_loader)
print("perplexity: ", np.exp(test_loss))

sen1 = ("Early in the pandemic, there was hope that the world would one day achieve herd immunity, "
"the point when the coronavirus lacks hosts to spread easily. But over a year later, the virus is " 
"crushing "
"India with a fearsome second wave and surging in countries from Asia to Latin America.")

sen2 = ("Early in the pandemic, there was hope that the world would one day achieve herd immunity, "
"the point when the coronavirus lacks hosts to spread easily. But over a year later, the virus is "
"dancing "
"India with a fearsome second wave and surging in countries from Asia to Latin America.")

sen3 = ("Early in the pandemic, there was hope that the world would one day achieve herd immunity, "
"the point when the coronavirus lacks hosts to spread easily. But over a year later, the virus is " 
"run "
"India with a fearsome second wave and surging in countries from Asia to Latin America.")

sen4 = ("Early in the pandemic, there was hope that the "
"cat "
" would one day achieve herd immunity, "
"the point when the coronavirus lacks hosts to spread easily. But over a year later, the virus is"
"run "
"India with a fearsome second wave and surging in countries from Asia to Latin America.")

sen5 = sen1.split()
random.shuffle(sen5)
sen5 = " ".join(sen5)

sen6 = " ".join(['Early in the']*10)

sen_list = [sen1, sen2, sen3, sen4, sen5, sen6]

for sen in sen_list:

    print(sen)
    #with open(PATH + "temp_sentence.txt", 'w') as text_file:
    #    print(sen, file = text_file)

    #temp_ds = torchtext.legacy.datasets.LanguageModelingDataset(path=PATH + 'temp_sentence.txt', text_field=TEXT)


    #sen_iter = torchtext.legacy.data.BPTTIterator(temp_ds, batch_size=BATCH_SIZE, device=DEVICE, 
    #                                              bptt_len=BPTT_LENGTH, repeat=False)
    
    sen_set = TextStreamer(sen.split(), MAX_VOCAB_SIZE, BPTT_LENGTH, vocabulary = train_set.get_vocab())
    sen_loader = DataLoader(sen_set, 1)
    sen_loss = evaluate(best_model, sen_loader)
    print("perplexity: ", np.exp(sen_loss))
    print()
