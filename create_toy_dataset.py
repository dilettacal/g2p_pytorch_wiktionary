#!/usr/bin/env python
# coding: utf-8

# # Wiktionary Parser
#
# Please put the dump file in `data/wiki`.
# The latest dump file can be downloaded from this URL: https://dumps.wikimedia.org/dewiktionary/latest/dewiktionary-latest-pages-articles.xml.bz2

# In[1]:


"""
Parser with MediaWiki XML utility

Installation: pip install mwxml
Repo: https://github.com/mediawiki-utilities/python-mwxml

"""
import argparse
import os
import time


VALID_MODE = ["g2p", "p2g", "p2p"]

def str2bool(v):
    # https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def convert(seconds):
    return time.strftime("%H:%M:%S", time.gmtime(seconds))



##### DIRECTORIES ######

DATA_DIR = "data"
WIKI = os.path.join(DATA_DIR, "wiki")
PREPROCESSED_FILE_DIR = os.path.join(DATA_DIR, "preprocessed")

##### FILES #####

# http://kaskade.dwds.de/gramophone/
DE_WIKTIONARY_DATA = "de-wiktionary.data.txt"  ### old file

# Self-preprocessed wiktionary dump
# dewiktionary-latest-pages-articles.xml
DUMP_FILE = [file for file in os.listdir(os.path.join("data", "wiki")) if file.lower().endswith(".xml")][0]

# This file stores the preprocessed pages from wiktioanry dump file
PAGES_ALL_LANG_FILE = "cleaned_pages_all_lang"  ### for txt output
PAGES_ONLY_GERMAN = "cleaned_pages_german"  ### for txt output

# In[28]:


DISCARD = ["ˈ", "ː", "ʔ", "̯", "̩", "ˌ", "͡ "]  ### Cleanes the phones


# ## Creation of toy datasets

# This method should be used to create toy sequences from a `csv`-file.
#
# Parameters:
# 1. `length`: defines the sequence length
# 2. `samples`: defines how many sample sequences should be created
# 3. `file`: the path to the csv file
# 4. `cols`: the name of the columns, where words should be retrieved from
# 5. `wiktionary`: A flag indicating if words/phonemes are taken from a parsed dump wiktionary file or not (used for the file name)

# In[37]:


import random
import numpy as np
import os
import pandas as pd

random.seed(a=5, version=2)


##### DIRECTORIES ######
DATA_DIR = os.path.abspath("data")
WIKI = os.path.join(DATA_DIR, "wiki")
PREPROCESSED_FILE_DIR = os.path.join(DATA_DIR, "preprocessed")

# In[39]:


flatten = lambda l: [item for sublist in l for item in sublist]


def get_data(file, cols, sep=","):
    df = pd.read_csv(file, sep=sep)
    # df1 = df[['a','b']]
    df = df[[cols[0], cols[1]]].astype(str)
    words = df[cols[0]].values
    ipas = df[cols[1]].values
    return words.tolist(), ipas.tolist()



def generate_toy_dataset(length=3, samples=0, file=os.path.join(PREPROCESSED_FILE_DIR, "wiki_corpus_de.csv"), cols=["word_lc", "phonemes"],
                         only_same_type="p2p"):

    src_matrix, target_matrix = [], []
    words, ipa_transcriptions = get_data(file=file, cols=cols)
    s = np.arange(len(words))  ## index array

    if samples > 0:
        print("Total samples:", samples)
        samples = len(words)  ## so many sequences as in the file
    else: samples = len(words)

    random_idx = [[random.choice(s) for i in range(length)] for j in range(samples)]  ## pick indices
    check = only_same_type in ["p2p"]

    if only_same_type and check:
        ### p == "Phoneme", "g" == "grapheme"
        if only_same_type == "p2p":
            print("Generating p2p file...")
            ### this generates a corpus of ipa_words w/o spaces and their corresponding ipa_words w/ spaces
            src_matrix = flatten(
                [[''.join([ipa_transcriptions[i].lower() for i in seq])] for seq in random_idx])  ## word matrix
            target_matrix = flatten(
                [[' '.join([ipa_transcriptions[i].lower() for i in seq])] for seq in random_idx])  ## ipa matrix
    else:
        #### This generates a corpus of normal sequences, srcs are words and targets are ipa transcriptions
        src_matrix = flatten([[' '.join([words[i].lower() for i in seq])] for seq in random_idx])  ## word matrix
        target_matrix = flatten(
            [[' '.join([ipa_transcriptions[i].lower() for i in seq])] for seq in random_idx])  ## ipa matrix

    import pandas as pd
    if only_same_type:
        save_file = "{}_toy_wiki_de-de_{}.csv".format(only_same_type, length)  #
    else:
        save_file = "toy_wiki_de-de_{}.csv".format(length)  #
    print("File name:", save_file)
    df = pd.DataFrame(zip(src_matrix, target_matrix), columns=["input", "target"])
    df.to_csv(os.path.join(DATA_DIR, "{}".format(save_file)), encoding="utf-8", sep="\t", index=False)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Wiktionary Parser and toy dataset creator')
    ### Embedding size ####ize of word embeddings
    parser.add_argument('--seq_len', type=int, default=20, help='Sequence length: How many units in the sentences'),
    ### Hidden size ####
    parser.add_argument('--mode', type=str, default="p2p", help='Mode: P2P, P2G, G2P available')

    parser.add_argument('--samples', type=int, default=0, help="How many data")

    args = parser.parse_args()

    seq_len = args.seq_len
    mode = args.mode.lower()
    samples = args.samples

    if not isinstance(samples, int):
        samples = 0

    print("Sequence length:", seq_len)

    file = "wiki_corpus_de.csv"
    print(os.path.join(PREPROCESSED_FILE_DIR, file))
    assert os.path.isfile(os.path.join(PREPROCESSED_FILE_DIR, file)), "Preprocessed csv wiktioanry file is missing!"

    generate_toy_dataset(seq_len, samples, file=os.path.join(PREPROCESSED_FILE_DIR, file), cols=["words", "clean_ipa"], only_same_type=mode)

    print("Files generated!")





