import nltk
import tools
import numpy as np
from text_preprocessing import *

import os
os.chdir("D:/M1_X/X_period2/introductionToTextMining/Project_textmining/nontop_report_and_code/code")


"""
This file contains functions to work with titles: clean title, get tfidf,
...
"""

def clean_title_1(df, stpwds, valid_tags, stemming=False, verbose=False):
    titles = []
    for i in range(len(df['title'])):
        try:
            titles.append(clean_text_1(df.ix[i, 'title'], stpwds, valid_tags, 
            			  stemming=stemming))
        except Exception as inst:
            print(i)
        if verbose and i % 2000 == 0:
            print("Processing " + str(i+1) + "th title...")
    return titles

def clean_title_2(df, stpwds, valid_tags, stemming=False, verbose=False):
    titles = []
    for i in range(len(df['title'])):
        try:
            titles.append(clean_text_2(df.ix[i, 'title'], stpwds, valid_tags, 
            			  stemming=stemming))
        except Exception as inst:
            print(inst.message)
        if verbose and i % 2000 == 0:
            print("Processing " + str(i+1) + "th title...")
    return titles

def clean_title_3(df, stpwds, valid_tags, stemming=False, verbose=False):
    titles = []
    for i in range(len(df['title'])):
        try:
            titles.append(clean_text_3(df.ix[i, 'title'], stpwds, valid_tags, 
            			  stemming=stemming))
        except Exception as inst:
            print(inst.message)
        if verbose and i % 2000 == 0:
            print("Processing " + str(i+1) + "th title...")
    return titles

def tfidf(tfidf_matrix, citation_links, verbose=False):
    similarity = np.empty(len(citation_links))
    #
    cnt = 0
    for citation in citation_links:
        first = np.array(tfidf_matrix[citation[0]].todense())
        second = np.array(tfidf_matrix[citation[1]].todense())
        similarity[cnt] = np.inner(first, second)

        cnt += 1
        if verbose and cnt % 10000 == 0:
            print("Processing " + str(cnt) + "th title...")
    return similarity