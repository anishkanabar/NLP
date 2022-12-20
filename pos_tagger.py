import torch
import torch.nn as nn
import torch.optim as optim
#from torchtext.legacy.data import Field, TabularDataset, BucketIterator, Iterator
import torchdata
from torchtext import data
from torchtext import datasets
from collections import Counter
from torchtext.vocab import Vocab, vocab
import numpy as np
import random
import time
import matplotlib.pyplot as plt
from torchtext import vocab

SEED = 53113
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

device = torch.device('cuda')

VECTORS_CACHE_DIR = './.vector_cache'

# Use the UDPOS dataset
train_iter = datasets.UDPOS(split='train')
for i, example in enumerate(train_iter):
    print(f'The sentence for Example {i} ---')
    print(example[0])
    print(f'The UD tags for Example {i} ---')
    print(example[1])
    print(f'The PTB tags for Example {i} ---')
    print(example[2])
    print()
    if i == 2: break

W = 1
WINDOW_SIZE = (2 * W + 1)

SENT_START_WORD = ''
SENT_END_WORD = ''
SENT_START_TAG = ''
SENT_END_TAG = ''

def add_sent_start_end(data_iter, w):
    for (words, ud_tags, ptb_tags) in data_iter:
        new_words = [SENT_START_WORD] * w + words + [SENT_END_WORD] * w
        new_ud_tags = [SENT_START_TAG] * w+ ud_tags + [SENT_END_TAG] * w
        ## MISSING PART: ADD YOUR CODE BELOW
        new_ptb_tags = [SENT_START_TAG] * w+ ptb_tags + [SENT_END_TAG] * w
        ## ADD YOUR CODE ABOVE
        yield(new_words, new_ud_tags, new_ptb_tags)
        
def create_windows(data_iter, w):
    window_size = 2*w + 1
    for (words, ud_tags, ptb_tags) in data_iter:
        words_zip = zip(*[words[i:] for i in range(window_size)])
        ud_zip = zip(*[ud_tags[i:] for i in range(window_size)])
        ptb_zip = zip(*[ptb_tags[i:] for i in range(window_size)])
        for word_sseq, ud_sseq, ptb_sseq in zip(
                words_zip, ud_zip, ptb_zip):
            yield(word_sseq, ud_sseq, ptb_sseq)
            
def preprocess_data_seq(data_iter, w):
    data_iter_a = add_sent_start_end(data_iter, w)
    data_iter_b = create_windows(data_iter_a, w)
    return data_iter_b

def test_preprocess_data_seq():
    train_iter_0 = datasets.UDPOS(split = 'train')    
    train_iter_demo = preprocess_data_seq(train_iter_0, 1)
    ex0 = (('', 'Al', '-'), 
           ('', 'PROPN', 'PUNCT'), 
           ('', 'NNP', 'HYPH'))
    ex1 = (('Al', '-', 'Zaman'), 
           ('PROPN', 'PUNCT', 'PROPN'), 
           ('NNP', 'HYPH', 'NNP'))
    ex2 = (('-', 'Zaman', ':'), 
           ('PUNCT', 'PROPN', 'PUNCT'), 
           ('HYPH', 'NNP', ':'))
    assert ex0 == next(train_iter_demo)
    assert ex1 == next(train_iter_demo)
    assert ex2 == next(train_iter_demo)
    print('Test passed.')
    
test_preprocess_data_seq()

# It is important to recreate the training set afresh 
# each time they are used, since they are Python iterators, and
# once use, cannot be reused.  One could, however, cast them into
# a list, which would store them permanently.

train_iter_0 = datasets.UDPOS(split='train')    
train_iter_vocab = preprocess_data_seq(train_iter_0, 1)

counter_words = Counter()
counter_ud = Counter()
counter_ptb = Counter()

for (text, pos_ud, pos_ptb) in train_iter_vocab:
    counter_words.update(text)
    counter_ud.update(pos_ud)
    counter_ptb.update(pos_ptb)

vocab_words = vocab(counter_words)    
vocab_ud = vocab(counter_ud)
vocab_ptb = vocab(counter_ptb)

from torch.utils.data import DataLoader
TAG = 'ud'

def collate_fn(batch, w = W, tag = TAG):
    ## WRITE YOUR CODE BELOW
    vocab_words_itos = vocab_words.get_itos()
    vocab_ud_itos = vocab_ud.get_itos()

    labels = torch.zeros(len(batch), dtype=torch.int64).to(device)
    word_idxs = torch.zeros(len(batch), len(batch[0][0]), dtype=torch.int64).to(device)
    for i in range(len(batch)):
        tag = batch[i][1][2]
        if tag in vocab_ud_itos:
            labels[i] = vocab_ud_itos.index(tag)
        for j in range(len(batch[i][0])):
            word = batch[i][0][j]
            if word in vocab_words_itos:
                word_idxs[i,j] = vocab_words_itos.index(word)
    return labels.to(device), word_idxs.cuda()

def test_collate():
    pos = [5, 6, 1, 4]  
    examples = []
    vocab_words_itos = vocab_words.get_itos()
    vocab_ud_itos = vocab_ud.get_itos()
    vocab_ptb_itos = vocab_ptb.get_itos()

    for perm in ['03022', '33210', '33211', '11101']:
        words = []
        utags = []
        ptags = []
        for ind in perm:
            ind = int(ind)
            words.append(vocab_words_itos[pos[ind]])
            utags.append(vocab_ud_itos[pos[ind]+1])
            ptags.append(vocab_ptb_itos[pos[ind]])
        examples.append((words, utags, ptags))
    lt =  torch.tensor([6, 2, 2, 7]).to(device)
    wt = torch.tensor([
        [5, 4, 5, 1, 1],
        [4, 4, 1, 6, 5],
        [4, 4, 1, 6, 6],
        [6, 6, 6, 5, 6]]).to(device)
    rlt, rwt = collate_fn(examples, w = 2)
    assert torch.equal(lt, rlt)
    assert torch.equal(wt, rwt)
    print("Test passed")

test_collate()

import torch.nn.functional as F

class NNPOSTagger(nn.Module):
    def __init__(self,
                 window_size,
                 vocab_size, 
                 embedding_dim, 
                 hidden_dim, 
                 output_dim,
                 nonlinearity, 
                 # These are used for later tasks
                 use_glove = False, 
                 freeze_glove = False):      
        super(NNPOSTagger, self).__init__()
        
        ## WRITE YOUR CODE BELOW 
        self.window_size = window_size
        self.embedding_dim = embedding_dim
        self.nonlinearity = nonlinearity
        self.Embedding = nn.Embedding(vocab_size, embedding_dim)
        self.Linear1 = nn.Linear(window_size * embedding_dim, hidden_dim)
        self.Linear2 = nn.Linear(hidden_dim, output_dim)
      
        
    def forward(self, word_idxs_batch):
        
        ## WRITE YOUR CODE BELOW.
        embeds = self.Embedding(word_idxs_batch).reshape(len(word_idxs_batch), self.window_size * self.embedding_dim)
        out = self.nonlinearity(self.Linear1(embeds))
        out = self.Linear2(out)
        log_probs = F.log_softmax(out, dim=1)
        return log_probs

model = NNPOSTagger(window_size = WINDOW_SIZE, 
                    vocab_size = len(vocab_words), 
                     embedding_dim = 300, 
                     hidden_dim = 128, 
                     output_dim = len(vocab_ud),
                     nonlinearity = nn.Tanh(), 
                     use_glove = False,
                     freeze_glove = False).to(device)

loss_function = torch.nn.NLLLoss()

def train_an_epoch(dataloader):
    ## WRITE YOUR CODE BELOW
    model.train() # Sets the module in training mode.
    log_interval = 500

    for idx, (label, text) in enumerate(dataloader):
        model.zero_grad()
        log_probs = model(text)
        loss = loss_function(log_probs, label)
        loss.backward()
        optimizer.step()

def get_accuracy(dataloader):
    model.eval()
    with torch.no_grad():    
        total_acc, total_count = 0, 0
        for idx, (label, word_idxs) in enumerate(dataloader):
            log_probs = model(word_idxs)
            total_acc += (log_probs.argmax(1) == label).sum().item()
            total_count += label.size(0)
    return total_acc/total_count

BATCH_SIZE = 64 # batch size for training
  
train_0, valid_0, test_0 = train_data_0 = datasets.UDPOS(
    split = ('train', 'valid', 'test'))
train_data = list(preprocess_data_seq(train_0, W))
valid_data = list(preprocess_data_seq(valid_0, W))
test_data = list(preprocess_data_seq(test_0, W))

train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE,
                              shuffle=True, 
                              collate_fn=collate_fn)
valid_dataloader = DataLoader(valid_data, batch_size=BATCH_SIZE,
                              shuffle=False, 
                              collate_fn=collate_fn)
test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE,
                             shuffle=False, 
                             collate_fn=collate_fn)

EPOCHS = 3 # epoch
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

accuracies=[]
for epoch in range(1, EPOCHS + 1):
    epoch_start_time = time.time()
    train_an_epoch(train_dataloader)
    accuracy = get_accuracy(valid_dataloader)
    accuracies.append(accuracy)
    time_taken = time.time() - epoch_start_time
    print(f'Epoch: {epoch}, time taken: {time_taken:.1f}s, validation accuracy: {accuracy:.3f}.')
    
plt.plot(range(1, EPOCHS+1), accuracies)

EPOCHS = 15 # epoch
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

for epoch in range(1, EPOCHS + 1):
    train_an_epoch(train_dataloader)

accuracy = get_accuracy(test_dataloader)
accuracy

glove = vocab.GloVe('6B',cache=VECTORS_CACHE_DIR)
glove_vectors = glove.get_vecs_by_tokens(vocab_words.get_itos())

import torch.nn.functional as F

class NNPOSTagger(nn.Module):
    def __init__(self,
                 window_size,
                 vocab_size, 
                 embedding_dim, 
                 hidden_dim, 
                 output_dim,
                 nonlinearity, 
                 # These are used for later tasks
                 use_glove = True, 
                 freeze_glove = False):      
        super(NNPOSTagger, self).__init__()

        self.window_size = window_size
        self.embedding_dim = embedding_dim
        self.nonlinearity = nonlinearity
        
        if use_glove:
            self.Embedding = nn.Embedding.from_pretrained(glove_vectors, freeze=freeze_glove)
        else:
            self.Embedding = nn.Embedding(vocab_size, embedding_dim)
        self.Linear1 = nn.Linear(window_size * embedding_dim, hidden_dim)
        self.Linear2 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, word_idxs_batch):
        
        embeds = self.Embedding(word_idxs_batch).reshape(len(word_idxs_batch), self.window_size * self.embedding_dim)
        out = self.nonlinearity(self.Linear1(embeds))
        out = self.Linear2(out)
        log_probs = F.log_softmax(out, dim=1)
        return log_probs

    model = NNPOSTagger(window_size = WINDOW_SIZE, 
                    vocab_size = len(vocab_words), 
                     embedding_dim = 300, 
                     hidden_dim = 128, 
                     output_dim = len(vocab_ud),
                     nonlinearity = nn.Tanh(), 
                     use_glove = True,
                     freeze_glove = True).to(device)

EPOCHS = 15 # epoch
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

accuracies=[]
for epoch in range(1, EPOCHS + 1):
    epoch_start_time = time.time()
    train_an_epoch(train_dataloader)
    accuracy = get_accuracy(valid_dataloader)
    accuracies.append(accuracy)
    time_taken = time.time() - epoch_start_time
    print(f'Epoch: {epoch}, time taken: {time_taken:.1f}s, validation accuracy: {accuracy:.3f}.')
    
plt.plot(range(1, EPOCHS+1), accuracies)

print(get_accuracy(test_dataloader))

model = NNPOSTagger(window_size = WINDOW_SIZE, 
                    vocab_size = len(vocab_words), 
                     embedding_dim = 300, 
                     hidden_dim = 128, 
                     output_dim = len(vocab_ud),
                     nonlinearity = nn.Tanh(), 
                     use_glove = True,
                     freeze_glove = False).to(device)

import time
import matplotlib.pyplot as plt

EPOCHS = 15 # epoch
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

accuracies=[]
for epoch in range(1, EPOCHS + 1):
    epoch_start_time = time.time()
    train_an_epoch(train_dataloader)
    accuracy = get_accuracy(valid_dataloader)
    accuracies.append(accuracy)
    time_taken = time.time() - epoch_start_time
    print(f'Epoch: {epoch}, time taken: {time_taken:.1f}s, validation accuracy: {accuracy:.3f}.')
    
plt.plot(range(1, EPOCHS+1), accuracies)

print(get_accuracy(test_dataloader))
