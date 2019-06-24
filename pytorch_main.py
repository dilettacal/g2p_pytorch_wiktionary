# -*- coding: utf-8 -*-

"""
Implementation is based on this tutorial:

How to build a Grapheme-to-Phoneme (G2P) model using PyTorch:
https://fehiepsi.github.io/blog/grapheme-to-phoneme/

"""
import datetime

import Levenshtein
import argparse
import random
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torchtext
from torch.autograd import Variable
from torchtext import data
import numpy as np
import pandas as pd
import torch.nn.functional as F
import os
import stat
import sys

VALID_MODE = ["p2p", "g2p", "p2g"]

DATA_DIR = os.path.abspath("data")
DATA_DIR_PREPRO = os.path.join(DATA_DIR, "preprocessed")
WIKI_DIR = os.path.join(DATA_DIR,  "wiki")



def check_permissions(start_dir):
    for root, dirs, files in os.walk(start_dir):
        for d in dirs:
            os.chmod(os.path.join(root, d), 0o777)
        for f in files:
            os.chmod(os.path.join(root, f), 0o777)


def permission_checker(dir):
    if sys.platform == "Windows" or sys.platform.lower().startswith("win"):
        os.chmod(os.path.abspath(dir), stat.S_IWRITE)  ### Windows problems on directory rights
    else:
        check_permissions(".")


"""

>>>> Script configuration
>>>> General configuration (CONFIG) are fixed.

They include:
 - the path to all the data, 
 - to the splittings (train.tsv, val.tsv, test.tsv)
 - max_len (the max number of unrolling steps during decoding at test time (when the target sequence should not be known)
 - attention: Use or not, default is True
 - Beam size: How many beams --> this is appleid only at test time
 - Log every: when results should be displayed
 - Lr_decay: Learning rate decay
 - Lr_MIN: the minimum value for learning rate. If this is reached and no improvements happened for 'n_bad_loss' times, then the training is left
 - n_bad_loss: patience for no validation improvements
 - clip: gradient clipping, default 2.3 (this is an arbitrary value)
 - cuda: if this should run on CUDA or not, if cuda not available, program runs automatically on CPU
 - intermediate_path: the root path where results from a training are stored in
"""

CONFIG = {
    'data_path': "data",
    'train_data': 'train.tsv',
    'valid_data': 'val.tsv',
    'test_data': 'test.tsv',
    'max_len': 35,  # max length of grapheme/phoneme sequences - used to unroll the decoder
    'beam_size': 5,  # size of beam for beam-search
    'attention': True,  # use attention or not
    'log_every': 100,  # number of iterations to log and validate training
    'lr_decay':  0.5,  # decay lr when not observing improvement in val_loss
    'lr_min': 1e-6,  # stop when lr is too low
    'n_bad_loss': 10,  # number of bad val_loss before decaying
    'clip': 2.3,  # clip gradient, to avoid exploding gradient
    'cuda': True,  # using gpu or not
    'seed': 5,  # initial seed
    'intermediate_path': 'results',  # path to save models
    'train_samples': 50000,
    'val_samples': 5000,
    'test_samples': 1000
}


"""
>>> RUNTIME CONFIGURATION

These configurations can be set up at runtime via CLI.
They all have default values, thus the program can also be ran with default values

"""

def get_args_parser():

    parser = argparse.ArgumentParser(description='P2G or G2P')
    ### Hidden size ####
    parser.add_argument('--emb', type=int, default=500, help='Mode: P2P, P2G, G2P, G2G available')

    parser.add_argument('--hid', type=int, default=500, help="hidden size")

    parser.add_argument('--epochs', type=int, default=10,
                        help="Epochs")

    parser.add_argument('--bs', type=int, default=100, help="Batch size")
    parser.add_argument('--lr', type=float, default=0.07, help="Learning rate")
    parser.add_argument('--file', type=str, default="p2p_toy_wiki_de-de_3.csv")

    parser.add_argument('--mode', type=str, default="p2p")

    return parser




"""
>>>> MODEL 
        Model class: handles both Encoder and Decoder
        Encoder class
        Decoder class
        Attention class --> Attention layer
"""


class Encoder(nn.Module):

    def __init__(self, vocab_size, d_embed, d_hidden):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_embed, padding_idx=1)
        self.lstm = nn.LSTMCell(d_embed, d_hidden)
        self.d_hidden = d_hidden

    def forward(self, x_seq, cuda=False):
        o = []
        e_seq = self.embedding(x_seq)  # seq x batch x dim
        tt = torch.cuda if cuda else torch  # use cuda tensor or not
        # create initial hidden state and initial cell state
        h = tt.FloatTensor(e_seq.size(1), self.d_hidden).zero_()
        c = tt.FloatTensor(e_seq.size(1), self.d_hidden).zero_()

        for e in e_seq.chunk(e_seq.size(0), 0):
            e = e.squeeze(0)
            h, c = self.lstm(e, (h, c))
            o.append(h)
        return torch.stack(o, 0), h, c

class Attention(nn.Module):
    """Dot global attention from https://arxiv.org/abs/1508.04025"""

    def __init__(self, dim):
        super(Attention, self).__init__()
        self.linear = nn.Linear(dim * 2, dim, bias=False)

    def forward(self, x, context=None):
        if context is None:
            return x
        assert x.size(0) == context.size(0)  # x: batch x dim
        assert x.size(1) == context.size(2)  # context: batch x seq x dim
        attn = F.softmax(context.bmm(x.unsqueeze(2)).squeeze(2), dim=1)
        weighted_context = attn.unsqueeze(1).bmm(context).squeeze(1)
        o = self.linear(torch.cat((x, weighted_context), 1))
        return torch.tanh(o)


class Decoder(nn.Module):
    def __init__(self, vocab_size, d_embed, d_hidden):
        super(Decoder, self).__init__()

        if vocab_size < d_embed:
            self.embedding = nn.Embedding(vocab_size, vocab_size, padding_idx=1)
            self.embedding.weight.data = torch.eye(vocab_size)
            self.lstm = nn.LSTMCell(vocab_size, d_hidden)
        else:
            self.embedding = nn.Embedding(vocab_size, d_embed, padding_idx=1)
            self.lstm = nn.LSTMCell(d_embed, d_hidden)
        self.attn = Attention(d_hidden)
        self.linear = nn.Linear(d_hidden, vocab_size)

    def forward(self, x_seq, h, c, context=None):
        o = []
        e_seq = self.embedding(x_seq)
        for e in e_seq.chunk(e_seq.size(0), 0):
            e = e.squeeze(0)
            h, c = self.lstm(e, (h, c))
            o.append(self.attn(h, context))
        o = torch.stack(o, 0)
        o = self.linear(o.view(-1, h.size(1)))
        return F.log_softmax(o, dim=1).view(x_seq.size(0), -1, o.size(1)), h, c


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.encoder = Encoder(config.g_size, config.d_embed,
                               config.d_hidden)
        self.decoder = Decoder(config.p_size, config.d_embed,
                               config.d_hidden)
        self.config = config

    def forward(self, input_seq, target_seq=None):
        o, h, c = self.encoder(input_seq, self.config.cuda)
        #context = o.t() if self.config.attention else None
        context = o.transpose(0,1) if self.config.attention else None
        if target_seq is not None:  # not generate
            return self.decoder(target_seq, h, c, context)
        else:
            assert input_seq.size(1) == 1  # make sure batch_size = 1
            return self._generate(h, c, context)

    def _generate(self, h, c, context):
        beam = Beam(self.config.beam_size, cuda=self.config.cuda)
        # Make a beam_size batch.
        h = h.expand(beam.size, h.size(1))
        c = c.expand(beam.size, c.size(1))
        context = context.expand(beam.size, context.size(1), context.size(2))

        for i in range(self.config.max_len):  # max_len = 20
            x = beam.get_current_state()
            o, h, c = self.decoder(Variable(x.unsqueeze(0)), h, c, context)
            if beam.advance(o.data.squeeze(0)):
                break
            h.data.copy_(h.data.index_select(0, beam.get_current_origin()))
            c.data.copy_(c.data.index_select(0, beam.get_current_origin()))
        tt = torch.cuda if self.config.cuda else torch
        return tt.LongTensor(beam.get_hyp(0))


"""
>>>> Model utilities    
    Beam class to perform beam search
    Phoneme Error Rate computation
    Learning rate decay method 

"""

# Based on https://github.com/MaximumEntropy/Seq2Seq-PyTorch/
class Beam(object):
    """Ordered beam of candidate outputs."""

    def __init__(self, size, pad=1, bos=2, eos=3, cuda=False):
        """Initialize params."""
        self.size = size
        self.done = False
        self.pad = pad
        self.bos = bos
        self.eos = eos
        self.tt = torch.cuda if cuda else torch

        # The score for each translation on the beam.
        self.scores = self.tt.FloatTensor(size).zero_()

        # The backpointers at each time-step.
        self.prevKs = []

        # The outputs at each time-step.
        self.nextYs = [self.tt.LongTensor(size).fill_(self.pad)]
        self.nextYs[0][0] = self.bos

    # Get the outputs for the current timestep.
    def get_current_state(self):
        """Get state of beam."""
        return self.nextYs[-1]

    # Get the backpointers for the current timestep.
    def get_current_origin(self):
        """Get the backpointer to the beam at this step."""
        return self.prevKs[-1]

    def advance(self, workd_lk):
        """Advance the beam."""
        num_words = workd_lk.size(1)

        # Sum the previous scores.
        if len(self.prevKs) > 0:
            beam_lk = workd_lk + self.scores.unsqueeze(1).expand_as(workd_lk)
        else:
            beam_lk = workd_lk[0]

        flat_beam_lk = beam_lk.view(-1)

        bestScores, bestScoresId = flat_beam_lk.topk(self.size, 0,
                                                     True, True)
        self.scores = bestScores

        # bestScoresId is flattened beam x word array, so calculate which
        # word and beam each score came from
        prev_k = bestScoresId / num_words
        self.prevKs.append(prev_k)
        self.nextYs.append(bestScoresId - prev_k * num_words)
        # End condition is when top-of-beam is EOS.
        if self.nextYs[-1][0] == self.eos:
            self.done = True
        return self.done

    def get_hyp(self, k):
        """Get hypotheses."""
        hyp = []
        # print(len(self.prevKs), len(self.nextYs), len(self.attn))
        for j in range(len(self.prevKs) - 1, -1, -1):
            hyp.append(self.nextYs[j + 1][k])
            #defence for beamsearch error
            if self.nextYs[j + 1][k] == self.eos:
                hyp=[self.eos]
            k = self.prevKs[j][k]
        return hyp[::-1]


### Util methods #####

def phoneme_error_rate(p_seq1, p_seq2):
    """
    Computes the phoneme error rate based on the utility Levenshtein
    :param p_seq1:
    :param p_seq2:
    :return:
    """
    p_vocab = set(p_seq1 + p_seq2)
    p2c = dict(zip(p_vocab, range(len(p_vocab))))
    c_seq1 = [chr(p2c[p]) for p in p_seq1]
    c_seq2 = [chr(p2c[p]) for p in p_seq2]
    return Levenshtein.distance(''.join(c_seq1),
                                ''.join(c_seq2)) / len(c_seq2)

def adjust_learning_rate(optimizer, lr_decay):
    for param_group in optimizer.param_groups:
        param_group['lr'] *= lr_decay


"""
TOKENIZATION, DATASETS AND VOCABULARIES

 >>>>  Methods to create vocabularies, handle datasets and tokenize sequences
"""


def wiki_tokenize(reverse=True):
    """
    Tokenize a given sequence x at CHAR level
    :param reverse: reverses the input (hallo --> ollah), see: https://arxiv.org/pdf/1409.3215.pdf
    :return: the tokenized sequence
    """
    discard = ["ˈ", "ː", "ʔ", "̯", "̩"]
    if reverse:
        return lambda x: list(x)[::-1] ## this reverses the input
    else:
        return lambda x: [el for el in list(x) if el not in discard]


######## Generate the tokenizers which will vectorize/numericalize the inputs for the model #######
def generate_vocab(init_token, eos_token=None, tokenize_fn=None):
    """
    Creates vocabularies for a given source.
    Vocabularies should be created for both source and target language.
    Field objects in PyTorch handles everything, from the creation of the vocabularies, to padding, numericalization etc.
    See:  `Field <https://torchtext.readthedocs.io/en/latest/data.html#field>`
    :param init_token: Begin of sequence token (e.g. ^), default: None
    :param eos_token: End of sequence token (e.g. $), default: None
    :param tokenize_fn: How the sequence is to be tokenized, default: split on blank spaces (str.split(" "))
    :return: a Field object to handle all operation
    """
    if eos_token:
        field = data.Field(init_token=init_token, eos_token=eos_token, tokenize=tokenize_fn)
    else:
        field = data.Field(init_token=init_token, tokenize=tokenize_fn)
    return field



class TSVDataset(torchtext.data.Dataset):

    def __init__(self, data_lines, src_field, trg_field):
        ### input and target are fixed labels used during the training ## changes here should also be applied in the training methods
        fields = [('input', src_field), ('target', trg_field)]
        self.examples = []
        for inp, trg in data_lines:
            self.examples.append(data.Example.fromlist([inp, trg], fields))

       # self.sort_key = lambda x: (len(x.src), len(x.target))
        self.sort_key = lambda x: (len(x.src))

        super(TSVDataset, self).__init__(examples=self.examples, fields=fields)

    def reduce_examples(self, num):
        self.examples = self.examples[:num]

    @classmethod
    def splits(cls, path, src_field, trg_field, root='',
               seed=5, revese_comb=False, fractions=[], **kwargs):
        if seed:
            random.seed(seed)
        df = pd.read_csv(path, encoding="utf-8", sep="\t")
        print("Total number of samples:", len(df))

        if revese_comb:
            ### this is used in case the model is trained for g2p or p2g task
            ### Values (units) are flipped on the horizzontal axis, while column names (input,target) remains unchanged
            df= pd.DataFrame(np.fliplr(df.values), columns=df.columns, index=df.index)

        if len(fractions) == 3:
            ### generate splits
            fractions = np.array(fractions)
        else:
            ### 60% training, 20% validation, 20% testing
            fractions = np.array([0.6, 0.2, 0.2])

        # shuffle your input
        df_to_split = df.sample(frac=1, random_state=seed)

        # split into 3 parts
        train, val, test = np.array_split(df_to_split, (fractions[:-1].cumsum() * len(df_to_split)).astype(int))

        train_lines = train[["input", "target"]].values.tolist()
        val_lines = val[["input", "target"]].values.tolist()
        test_lines = test[["input", "target"]].values.tolist()

        train_data = cls(train_lines, src_field, trg_field)
        val_data = cls(val_lines, src_field, trg_field)
        test_data = cls(test_lines, src_field, trg_field)
        return (train_data, val_data, test_data)



### global variables
ITERATION = n_total = train_loss = n_bad_loss = 0
BEST_VAL_LOSS = 10
INIT_TIME = time.time()


"""
Training, validation and testing methods

"""
def train(config, train_iter,val_iter, model, criterion, optimizer, epoch, logger):
    global ITERATION, n_total, train_loss, n_bad_loss
    global INIT_TIME, BEST_VAL_LOSS

    logger.log("=> EPOCH {}".format(epoch))
    train_iter.init_epoch()
    for batch in train_iter:
        ITERATION += 1
        model.train()

        output, _, __ = model(batch.input, batch.target[:-1].detach())
        target = batch.target[1:]
        loss = criterion(output.view(output.size(0) * output.size(1), -1),
                         target.view(target.size(0) * target.size(1)))

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip, 'inf')
        optimizer.step()

        n_total += batch.batch_size
        train_loss += loss.item() * batch.batch_size

        if ITERATION % config.log_every == 0:
            train_loss /= n_total
            val_loss = validate(val_iter, model, criterion)
            logger.log("   % Time: {:5.0f} | Iteration: {:5} | Batch: {:4}/{}"
                  " | Train loss: {:.4f} | Val loss: {:.4f}"
                  .format(time.time() - INIT_TIME, ITERATION, train_iter.iterations,
                          len(train_iter), train_loss, val_loss))

            # test for val_loss improvement
            n_total = train_loss = 0
            if val_loss < BEST_VAL_LOSS:
                BEST_VAL_LOSS = val_loss
                n_bad_loss = 0
                torch.save(model.state_dict(), config.best_model)
            else:
                n_bad_loss += 1
            if n_bad_loss == config.n_bad_loss:
                BEST_VAL_LOSS = val_loss
                n_bad_loss = 0
                adjust_learning_rate(optimizer, config.lr_decay)
                new_lr = optimizer.param_groups[0]['lr']
                logger.log("=> Adjust learning rate to: {}".format(new_lr))
                if new_lr < config.lr_min:
                    return True
    return False


def validate(val_iter, model, criterion):
 #   print("Running validation on validation set.....")
    model.eval()
    val_loss = 0
    val_iter.init_epoch()
    for batch in val_iter:
        output, _, __ = model(batch.input, batch.target[:-1])
        target = batch.target[1:]
        loss = criterion(output.squeeze(1), target.squeeze(1))
        val_loss += loss.item()* batch.batch_size
    return val_loss / len(val_iter.dataset)


def test(test_iter, model, logger):
    logger.log("Testing the model on the test dataset. \nDataset length: {}".format(len(test_iter)))
    model.eval()
    test_per = 0
    #print("Test dataset contains {} batches of length {}.".format(str(len(test_iter)), str(len(next(iter(test_iter))))))
    for i, batch in enumerate(test_iter):
        output = model(batch.input).data.tolist()
        target = batch.target[1:].squeeze(1).data.tolist()
        per = phoneme_error_rate(output, target)
        test_per += per
    test_per = test_per / len(test_iter.dataset) * 100
    logger.log("Phoneme error rate (PER): {:.2f}".format(test_per))
    return test_per

def show(batch, model, logger):
    assert batch.batch_size == 1
    g_field = batch.dataset.fields['input']
    p_field = batch.dataset.fields['target']
    prediction = model(batch.input).data.tolist()[:-1]
    inputs = batch.input.squeeze(1).data.tolist()[1:][::-1] ## if input are reversed!
    targets = batch.target.squeeze(1).data.tolist()[1:-1]
    logger.log("SRC: \t {}\nTRG: \t {}\nPRED:\t {}\n".format(
        ''.join([g_field.vocab.itos[g] for g in inputs]),
        ' '.join([p_field.vocab.itos[p] for p in targets]),
        ' '.join([p_field.vocab.itos[p] for p in prediction])))



"""
Logger class 
"""


class Logger():
    '''Prints to a log file and to standard output'''
    def __init__(self, path, file_name="log.log", mode = "a"):
        if os.path.exists(path):
            self.path = path
            self.file_name = file_name
            self.mode = mode
        else:
            raise Exception('path does not exist')

    def log(self, info, stdout=True):
        with open(os.path.join(self.path, self.file_name), self.mode) as f:
            print(info, file=f)
        if stdout:
            print(info)

"""
MAIN METHOD TO RUN THE SCRIPT
"""
def main():

    fixed_config = argparse.Namespace(**CONFIG)

    cli_config = get_args_parser().parse_args()

    TRAIN_MODE = cli_config.mode.lower()
    assert TRAIN_MODE in VALID_MODE, "Please select right training mode (p2p, p2g, g2p)"

    FILE_NAME = cli_config.file.lower()

    reverse_combi= False

    assert TRAIN_MODE == FILE_NAME.lower().split("_")[0], "Please provide right file for training mode" + TRAIN_MODE


    if TRAIN_MODE == "p2g":
        #### files always store graphemes and phonemes
        reverse_combi = True


    #### Reading configuration from CLI #####

    print(cli_config.file)

    seq_len = int(cli_config.file.lower().split("_")[-1].split(".")[0])
    emb_dim = cli_config.emb
    hid_dim = cli_config.hid
    epochs = cli_config.epochs

    fixed_config.cuda = fixed_config.cuda and torch.cuda.is_available() ### setup training on cuda

    os.makedirs((fixed_config.intermediate_path), exist_ok=True)
    fixed_config.intermediate_path = os.path.join((fixed_config.intermediate_path), "pytorch")

    #### This is only used, because of permission rights errors ######
    permission_checker((fixed_config.intermediate_path))
    if not os.path.isdir((fixed_config.data_path)):
        raise print("error")

    ##### Setup seeds for the whole program #####
    torch.manual_seed(fixed_config.seed)
    if fixed_config.cuda:
        torch.cuda.manual_seed(fixed_config.seed)

    ### special tokens ####
    init_token = "^"
    eos_token = "$"

    ##### Vocabulary creation #####
    INPUT = generate_vocab(init_token=init_token, tokenize_fn=wiki_tokenize(reverse=True)) ###inputs are reversed!
    TARGET = generate_vocab(init_token=init_token, eos_token=eos_token, tokenize_fn=wiki_tokenize(False))

    print("Creating dataset and iterators...")


    ### Now datasets are created: the TSVDataset class generates splits (fractions: 60-20-20)
    train_data, val_data, test_data = TSVDataset.splits(path=os.path.join(fixed_config.data_path, FILE_NAME),
                                                        src_field=INPUT, trg_field=TARGET,
                                                        seed=fixed_config.seed, revese_comb=reverse_combi)


    #### reducing number of examples ####
    train_data.reduce_examples(fixed_config.train_samples) ## this takes 'train_samples' examples from the training data (which is 60% of the whole dataset)
    val_data.reduce_examples(fixed_config.val_samples)
    test_data.reduce_examples(fixed_config.test_samples)


    ##### Vocabularies are built on the provided train, val and test data
    #### These creates the mapping index > word, word > index and applies padding when train_data, val_data, test_data are passed to an iterator

    INPUT.build_vocab(train_data, val_data, test_data)
    TARGET.build_vocab(train_data, val_data, test_data)

    device = "cuda" if fixed_config.cuda else "cpu"

    #### Iterators ###

    train_iter = data.BucketIterator(train_data, batch_size=cli_config.bs,
                                     repeat=False, device=device, sort_key=lambda x: (len(x.input), len(x.target))) #creates buckets
    ### Iterator class if batch size == 1
    val_iter = data.Iterator(val_data, batch_size=1,
                             train=False, sort=False, device=device, sort_key=lambda x: (len(x.input), len(x.target)))
    test_iter = data.Iterator(test_data, batch_size=1,
                              train=False, device=device, sort=True, sort_key=lambda  x: (len(x.input), len(x.target)))


    fixed_config = fixed_config
    fixed_config.g_size = len(INPUT.vocab)
    fixed_config.p_size = len(TARGET.vocab)
    fixed_config.d_embed = emb_dim
    fixed_config.d_hidden = hid_dim
    fixed_config.lr = cli_config.lr
    store_path = (fixed_config.intermediate_path)
    store_path = os.path.join(store_path, TRAIN_MODE, str(seq_len), datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
    fixed_config.best_model = os.path.join(store_path)
    logger_path = fixed_config.best_model
    os.makedirs(fixed_config.best_model, exist_ok=True)
    permission_checker(fixed_config.best_model)
    fixed_config.best_model = os.path.join(fixed_config.best_model, "{}_{}.pth".format(TRAIN_MODE, seq_len))
    LR = fixed_config.lr
    print(LR)


    experiment_logger = Logger(path=logger_path, file_name="experiment.log")
    results_logger = Logger(path=logger_path, file_name="results.log")

   # experiment_logger.log("Type: {}".format(TRAIN_MODE))

    ###### Model creation #####

    model = Model(fixed_config)
    experiment_logger.log("Type: {}".format(TRAIN_MODE))
    experiment_logger.log("File: {}".format(FILE_NAME))
    experiment_logger.log("src_vocab: {}".format(len(INPUT.vocab)))
    experiment_logger.log("trg_vocab: {}".format(len(TARGET.vocab)))
    experiment_logger.log("enc_emb_dim: {}".format(model.encoder.embedding.embedding_dim))
    experiment_logger.log("dec_emb_dim: {}".format(model.decoder.embedding.embedding_dim))
    experiment_logger.log("hid_dim: {}".format(hid_dim))
    experiment_logger.log("Optimizer: adam")
    experiment_logger.log("LR: {}".format(LR))
    experiment_logger.log("Batch size: {}".format(cli_config.bs))
    experiment_logger.log("Epochs: {}".format(cli_config.epochs))
    experiment_logger.log("Train samples: {}".format(len(train_data)))
    experiment_logger.log("Validation samples: {}".format(len(val_data)))
    experiment_logger.log("Test samples: {}".format(len(test_data)))
    experiment_logger.log("Model overview: \n{}".format(model))

    criterion = nn.NLLLoss()
    if fixed_config.cuda:
        model.cuda()
        criterion.cuda()
    #optimizer = optim.Adam(model.parameters(), lr=LR)  # use Adagrad
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)

    if True:
        for epoch in range(1, epochs+1):
            stop = train(fixed_config, train_iter,val_iter, model, criterion, optimizer, epoch, experiment_logger)
            if stop:
                break

    ################ Running the model on the test set ############
    print("Load model:")
    model.load_state_dict(torch.load(fixed_config.best_model))

    #### Test with PER computation
    test(test_iter, model, experiment_logger)

    #test_iter.init_epoch()
    ### Generating translations
    for i, batch in enumerate(test_iter):
        show(batch, model, results_logger)
        if i == 100:
            break


if __name__ == '__main__':
    main()

