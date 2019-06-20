#!/usr/bin/env python
# coding: utf-8

# # Wiktionary Parser
#
# Please put the dump file in `data/wiki`.
# The latest dump file can be downloaded from this URL: https://dumps.wikimedia.org/dewiktionary/latest/dewiktionary-latest-pages-articles.xml.bz2



"""
Parser with MediaWiki XML utility

Installation: pip install mwxml
Repo: https://github.com/mediawiki-utilities/python-mwxml

"""

import re
import stat
import time

import mwxml
from mwxml import Page

import random
import os
import pandas as pd

from pytorch_main import get_full_path

random.seed(a=5, version=2)

##### DIRECTORIES ######

DATA_DIR = get_full_path("data")
WIKI = os.path.join(DATA_DIR, "wiki")
PREPROCESSED_FILE_DIR = os.path.join(DATA_DIR, "preprocessed")

VALID_MODE = ["g2p", "p2g", "p2p", "g2g"]


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

DISCARD = ["ˈ", "ː", "ʔ", "̯", "̩", "ˌ", "͡ "]  ### Cleanes the phones


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



class WiktPage(object):
    def __init__(self, page: Page):
        self.id = page.id
        self.title = page.title
        self.revisions = [rev.text for rev in page]
        self.lang = None
        self.IPA = None
        self.syllabication = None  ### not tracked

    def __repr__(self):
        return f"Page_id = {self.id}, IPA = {self.IPA}"

    def set_ipa(self, ipa_string):
        self.IPA = ipa_string

    def __dict__(self):
        return {"id": self.id, "title": self.title, "IPA": self.IPA, "lang": self.lang}

    def __str__(self):
        return self.__dict__



def remove_tags(text):
    """
    Removes HTML code from a string, e.g. "<!--Spezialfall	NICHTlöschen-->"
    :param text:
    :return:
    """
    TAG_RE = re.compile(r'<[^>]+>')
    text = TAG_RE.sub('', text)
    return text



def persist_to_csv(page_list, out_file="german"):
    words = [page.title for page in page_list]
    ipa_transcriptions = [page.IPA for page in page_list]
    cleaned_transcriptions = [''.join(token for token in page.IPA if token not in DISCARD) for page in page_list]

    d = {"words": words, "ipa": ipa_transcriptions, "clean_ipa": cleaned_transcriptions}

    df = pd.DataFrame(d)
    file_name = "wiki_corpus_full.csv" if out_file == "all" else "wiki_corpus_de.csv"
    #### file is stored in data/preprocessed directory
    df.to_csv(os.path.join(PREPROCESSED_FILE_DIR, file_name), mode="w", encoding="utf-8", index=0)


def read_pages_from_dump(dump_file):
    """
    Reads all pages from a wikipedia dump file
    :param dump_file: wiktionary file as xml
    :return: a generator of WiktPage objects
    """
    for i, page in enumerate(dump_file.pages):
        wiktpage = WiktPage(page)
        yield wiktpage


"""
Wiktionary parse method
"""

def parse_wiktionary():
    #### directory configurations
    sys_name = os.name.lower()
    if "windows" in sys_name or "nt" in sys_name:
        print("Running on a Windows machine.")
        os.chmod(".", stat.S_IWRITE)  ## write problem on Windows
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

    persist_to_csv(valid_pages, out_file="all")
    ## Save  only german pages - 497265
    valid_pages = [page for page in valid_pages if page.lang == "german" or page.lang == "deutsch"]

    print("Total pages with 'German' language: %s" % str(len(valid_pages)))
    print("Total pages with 'None' revisions: %s" % str(len(revisions_none)))
    print("Persisting kept pages....")
    persist_to_csv(valid_pages, out_file="german")
    print("Pages persisted!")


flatten = lambda l: [item for sublist in l for item in sublist]

def get_data(file, cols, gr=False, sep="\t"):
    df = pd.read_csv(file, sep=sep)
    print(df.columns)
    # df1 = df[['a','b']]
    df = df[[cols[0], cols[1]]].astype(str)
    words = df[cols[0]].values
    ipas = df[cols[1]].values
    return words.tolist(), ipas.tolist()


if __name__ == '__main__':

    print("Processing started...")
    start = time.time()
    parse_wiktionary()
    end = time.time()
    print("Time: {}".format(convert(end - start)))
