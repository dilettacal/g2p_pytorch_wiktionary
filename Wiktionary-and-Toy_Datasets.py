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
import pickle
import re
import stat
import pandas as pd
import time

import mwxml
from mwxml import Page

VALID_MODE = ["g2p", "p2g", "p2p", "g2g"]


def str2bool(v):
    #https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


# In[26]:


def convert(seconds):
    return time.strftime("%H:%M:%S", time.gmtime(seconds))


# ## CONFIGURATIONS

# In[27]:


##### DIRECTORIES ######

DATA_DIR = "data"
WIKI = os.path.join(DATA_DIR, "wiki")
PREPROCESSED_FILE_DIR = os.path.join(DATA_DIR, "preprocessed")

##### FILES #####

#http://kaskade.dwds.de/gramophone/
DE_WIKTIONARY_DATA = "de-wiktionary.data.txt" ### old file

#Self-preprocessed wiktionary dump
# dewiktionary-latest-pages-articles.xml
DUMP_FILE = [file for file in os.listdir(os.path.join("data", "wiki")) if file.lower().endswith(".xml")][0]

#This file stores the preprocessed pages from wiktioanry dump file
PAGES_ALL_LANG_FILE = "cleaned_pages_all_lang" ### for txt output
PAGES_ONLY_GERMAN = "cleaned_pages_german" ### for txt output


# In[28]:


DISCARD = ["ˈ", "ː", "ʔ", "̯", "̩", "ˌ", "͡ "] ### Cleanes the phones



# ## Wiktionary Page object
# 
# The `WiktPage` object defines a Wiktionary Page from the dump file. Pages are processed by the `mwxml` parser. For each page, the parser  parses following information:
# - Page id
# - Title
# - Revisions
# 
# The field revision is a very long string containing all the page texts. This includes IPA information, as well.
# 
# IPA information begins with **{{Sprache|**. The used symbols "{{" and the label "Sprache" may differ for other language dumps!
# 
# The method `parse_wiktionary` is the main method used for parsing the whole wiktionary dump. In this method, WiktPage-objects are created and IPA information are extracted and saved in the `IPA` member. The program also stores the language (`lang`), because some items in the wiktionary may refer to foreign words. Some of these items may be empty or incomplete!

# In[30]:


class WiktPage(object):
    def __init__(self, page: Page):
        self.id = page.id
        self.title = page.title
        self.revisions = [rev.text for rev in page]
        self.lang = None
        self.IPA = None
        self.syllabication = None ### not tracked

    def __repr__(self):
        return f"Page_id = {self.id}, IPA = {self.IPA}"

    def set_ipa(self, ipa_string):
        self.IPA = ipa_string

    def __dict__(self):
        return {"id": self.id, "title": self.title, "IPA": self.IPA, "lang": self.lang}

    def __str__(self):
        return self.__dict__


# In[31]:


def remove_tags(text):
    """
    Removes HTML code from a string, e.g. "<!--Spezialfall	NICHTlöschen-->"
    :param text:
    :return:
    """
    TAG_RE = re.compile(r'<[^>]+>')
    text = TAG_RE.sub('', text)
    return text


# In[32]:


def persist_to_csv(page_list, out_file="german"):
    words = [page.title for page in page_list]
    ipa_transcriptions = [page.IPA for page in page_list]
    cleaned_transcriptions = [''.join(token for token in page.IPA if token not in DISCARD) for page in page_list]
    
    d = {"words": words, "ipa": ipa_transcriptions, "clean_ipa": cleaned_transcriptions}
    
    df = pd.DataFrame(d)
    file_name = "wiki_corpus_full.csv" if out_file=="all" else "wiki_corpus_de.csv"
    df.to_csv(os.path.join(PREPROCESSED_FILE_DIR, file_name), mode="w", encoding="utf-8", index=0)


# In[33]:


### to be used if output file should be a txt file!
def persist_pages(pages_list, delete_content=True, out_file=PAGES_ALL_LANG_FILE):
    if not os.path.isdir(os.path.join(".", PREPROCESSED_FILE_DIR)):
        os.makedirs(os.path.join(".", PREPROCESSED_FILE_DIR))

    file = os.path.join(".", PREPROCESSED_FILE_DIR, out_file)

    if delete_content:
        # Delete content
        raw = open(file, "w")
        raw.close()

        if out_file == PAGES_ALL_LANG_FILE:
            with open(file+".txt", 'a', encoding="utf-8") as f:
                print("File:", file)
                for i, page in enumerate(pages_list):
                    cleaned_ipa = ''.join(token for token in page.IPA if token not in DISCARD)
                    f.write(page.title + "\t" + page.IPA + "\t" + cleaned_ipa + "\t" + page.lang + "\n")
        else:
            with open(file+".txt", 'a', encoding="utf-8") as f:
                print("File:", file)
                for i, page in enumerate(pages_list):
                    cleaned_ipa = ''.join(token for token in page.IPA if token not in DISCARD)
                    f.write(page.title + "\t" + page.IPA + "\t" + cleaned_ipa + "\n")


# In[34]:


def read_pages_from_dump(dump_file):
    """
    Reads all pages from a wikipedia dump file
    :param dump_file: wiktionary file as xml
    :return: a generator of WiktPage objects
    """
    for i, page in enumerate(dump_file.pages):
        wiktpage = WiktPage(page)
        yield wiktpage


# ### Main parsing method

# In[35]:


### Main method!
def parse_wiktionary(csv=True):
    #### directory configurations
    sys_name = os.name.lower() 
    if "windows" in sys_name or "nt" in sys_name:
        print("Running on a Windows machine.")
        os.chmod(".", stat.S_IWRITE) ## write problem on Windows
    if not os.path.isdir(os.path.join(".", WIKI)):
        print("Directory created!")
        os.makedirs(os.path.join(".", WIKI))

    file = os.path.join(".", WIKI, DUMP_FILE)
    dump = mwxml.Dump.from_file(file)
    processed_pages = list(read_pages_from_dump(dump))

    print("Selecting pages to keep...")
    valid_pages = []
    revisions_none = []
    for i, page in enumerate(processed_pages):
        revisions = page.revisions  # all revisions are saved into a list
        text = revisions[0]
        if text is None:
            revisions_none.append(page)
            continue
        else:
            try:
                # Get language
                language = text.split("({{Sprache|")[1].split("}}")[0]
                page.lang = str(language).lower()
            except IndexError:
                page.lang = None
                pass

            # Retrieve IPA
            text = remove_tags(str(text))
            ipa = None
            try:
                # Given: {{Lautschrift|ˈsmr̩̂tan}}
                # 1st: {{Lautschrift|, ˈsmr̩̂tan}}
                # 2nd: ˈsmr̩̂tan, }} --> ˈsmr̩̂tan
                ipa = text.split("{{Lautschrift|")[1].split("}}")[0]
                page.set_ipa(ipa)
                valid_pages.append(page)
            except IndexError:
                page.set_ipa(None)

    print("Page cleaning done... ")

    ## Keep only pages with ipa and language != None - 601290
    valid_pages = [page for page in valid_pages if page.IPA and page.lang]
    print("Total pages - all languages: %s" % str(len(valid_pages)))
    if not csv:
        persist_pages(valid_pages, out_file=PAGES_ALL_LANG_FILE)
    else:
        persist_to_csv(valid_pages, out_file="all")
    ## Save  only german pages - 497265
    valid_pages = [page for page in valid_pages if page.lang == "german" or page.lang == "deutsch"]

    print("Total pages with 'German' language: %s" % str(len(valid_pages)))
    print("Total pages with 'None' revisions: %s" % str(len(revisions_none)))
    print("Persisting kept pages....")
    if not csv:
        persist_pages(valid_pages, out_file=PAGES_ONLY_GERMAN)
    else:
        persist_to_csv(valid_pages, out_file="german")
    print("Pages persisted!")


# In[36]:




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


# In[38]:


##### DIRECTORIES ######

DATA_DIR = "data"
WIKI = os.path.join(DATA_DIR, "wiki")
PREPROCESSED_FILE_DIR = os.path.join(DATA_DIR, "preprocessed")


# In[39]:


flatten = lambda l: [item for sublist in l for item in sublist]


# In[45]:


def get_data(file, cols,  gr=False, sep="\t"):
    df = pd.read_csv(file, sep=sep)
    print(df.columns)
    #df1 = df[['a','b']]
    df = df[[cols[0], cols[1]]].astype(str) 
    words = df[cols[0]].values
    ipas = df[cols[1]].values
    return words.tolist(), ipas.tolist()


# In[47]:


def generate_toy_dataset(length=3, samples=50000, file ="data/phonemes-de-de.csv", cols=["word_lc", "phonemes"], wiktionary=True, sep="\t", 
                         only_same_type="p"):
    
    src_matrix, target_matrix = [], []
    words, ipa_transcriptions = get_data(file=file, cols=cols, sep=sep)
    s = np.arange(len(words)) ## index array
    
    print(words[:10])
    
    if samples > 0:
        print("Total samples:", samples)
        samples = len(words) ## so many sequences as in the file
    
    random_idx = [[random.choice(s) for i in range(length)] for j in range(samples)]  ## pick indices
    check = only_same_type in ["p", "g"]

    if only_same_type and check:
            ### p == "Phoneme", "g" == "grapheme"
        if only_same_type== "p":
            ### this generates a corpus of ipa_words w/o spaces and their corresponding ipa_words w/ spaces
            src_matrix = flatten([[''.join([ipa_transcriptions[i].lower() for i in seq])] for seq in random_idx]) ## word matrix    
            target_matrix = flatten([[' '.join([ipa_transcriptions[i].lower() for i in seq])] for seq in random_idx]) ## ipa matrix
        
        if only_same_type== "g":
            ### this generates a corpus of words w/o spaces and their corresponding words w/ spaces
            src_matrix = flatten([[''.join([words[i].lower() for i in seq])] for seq in random_idx]) ## word matrix    
            target_matrix = flatten([[' '.join([words[i].lower() for i in seq])] for seq in random_idx]) ## ipa matrix
    else:
        #### This generates a corpus of normal sequences, srcs are words and targets are ipa transcriptions
        src_matrix = flatten([[' '.join([words[i].lower() for i in seq])] for seq in random_idx]) ## word matrix    
        target_matrix = flatten([[' '.join([ipa_transcriptions[i].lower() for i in seq])] for seq in random_idx]) ## ipa matrix
    
   

    import pandas as pd   
    
    if not wiktionary:
        if only_same_type:
            save_file = "{}_toy_de_de{}.csv".format(only_same_type, length)#
        else: save_file = "toy_de_de{}.csv".format(length)#
    else:
        if only_same_type:
             save_file = "{}_toy_wiki_de-de{}.csv".format(only_same_type, length)#
        else: save_file = "toy_wiki_de-de{}.csv".format(length)#
    print("File name:", save_file)
    df = pd.DataFrame(zip(src_matrix, target_matrix), columns=["input", "target"])
    df.to_csv(path_or_buf="data/{}".format(save_file), encoding="utf-8", sep="\t", index=False)


# In[49]:

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Wiktionary Parser and toy dataset creator')
    ### Embedding size ####ize of word embeddings
    parser.add_argument('--seq_len', type=int, default=5, help='Sequence length: How many units in the sentences'),
    ### Hidden size ####
    parser.add_argument('--mode', type=str, default="p2p", help='Mode: P2P, P2G, G2P, G2G available')

    parser.add_argument('--samples', type=int, default=50000, help="How many data")

    parser.add_argument('--parse', type=str2bool, default="False",
                        help="Parse Wiktionary XML files to generate a corpus of words and IPA transcriptions. Default: False")

    args = parser.parse_args()

    seq_len = args.seq_len
    mode = args.mode.lower()
    samples = args.samples
    parse = args.parse

    if parse:
        print("Processing started...")
        start = time.time()
        parse_wiktionary(csv=True)
        end = time.time()
        print("Total duration: {}".format(convert(end - start)))

    if not isinstance(samples, int):
        samples = None


    print("Sequence length:", seq_len)

    file = "wiki_corpus_de.csv"
    generate_toy_dataset(seq_len, samples, file=os.path.join(PREPROCESSED_FILE_DIR, file), cols=["words", "clean_ipa"], sep=",", only_same_type=mode)
    print("Files generated!")


# In[ ]:




