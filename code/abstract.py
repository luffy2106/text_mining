import nltk
import tools
import numpy as np
from text_preprocessing import *


import os
os.chdir("D:/M1_X/X_period2/introductionToTextMining/Project_textmining/nontop_report_and_code/code")

"""
This file contains functions to work with abstracts: clean abstract, get tfidf,
...
"""

# def abstract_to_keywords(abstract, w, stpwds, pos_filtering=True,
# 					     stemming=True):
#     tokens = abstract_tokenizer(abstract, stpwds)
#     if len(tokens) < w:
#         return tokens
#     else:
#         g = tools.term_to_graph(tokens, w)
#         return tools.get_keywords(g)

def clean_abstract_1(df, stpwds, valid_tags, stemming=False, verbose=False):
    abstracts = []
    for i in range(len(df['abstract'])):
        abstracts.append(clean_text_1(df['abstract'][i], stpwds, valid_tags, 
        				 stemming=stemming))
        if verbose and i % 2000 == 0:
            print("Processing " + str(i+1) + "th abstract...")
    return abstracts

def clean_abstract_2(df, stpwds, valid_tags, stemming=False, verbose=False):
    abstracts = []
    for i in range(len(df['abstract'])):
        abstracts.append(clean_text_2(df['abstract'][i], stpwds, valid_tags, 
        				 stemming=stemming))
        if verbose and i % 2000 == 0:
            print("Processing " + str(i+1) + "th abstract...")
    return abstracts

def clean_abstract_3(df, stpwds, valid_tags, stemming=False, verbose=False):
    abstracts = []
    for i in range(len(df['abstract'])):
        abstracts.append(clean_text_3(df['abstract'][i], stpwds, valid_tags, 
        				 stemming=stemming))
        if verbose and i % 2000 == 0:
            print("Processing " + str(i+1) + "th abstract...")
    return abstracts

def word_embedding(citation_links, abstracts, wm_model, verbose=False):
    dist = np.empty(len(citation_links))
    # 
    cnt = 0
    for citation in citation_links:
        dist[cnt] = wm_model.wmdistance(abstracts[citation[0]].split(),
                                       abstracts[citation[1]].split())
        cnt += 1
        if verbose and cnt % 10000 == 0:
            print("Processing " + str(cnt) + "th lines...")
    return dist

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
            print("Processing " + str(cnt) + "th citation link...")

    return similarity

