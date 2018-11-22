from __future__ import division
import numpy as np

import os
os.chdir("D:/M1_X/X_period2/introductionToTextMining/Project_textmining/nontop_report_and_code/code")


def mul_sparse_vector(veca, vecb):
    product = 0
    runa = 0
    runb = 0
    while runa < len(veca) and runb < len(vecb):
        if veca[runa][0] == vecb[runb][0]:
            product += veca[runa][1] * vecb[runb][1]
            runa += 1
            runb += 1
        elif veca[runa][0] < vecb[runb][0]:
            runa += 1
        else:
            runb += 1
    return product


def log_tf(tf):
    return 1 + np.log(tf)

def log_concav(document_length, tf):
    return 1 + np.log(1 + np.log(tf))

def k_concav(k, document_length, tf):
    return (k + 1 + tf)/(k + tf)

def p_tf(b, avd_length, document_length, tf):
    return tf/(1 - b + b * document_length/avd_length)

def theta_tf(theta, document_length, tf):
    if tf > 0:
        return tf + theta
    else:
        return 0

def k_p(k, b, avd_length, document_length, tf):
    return k_concav(k, document_length, 
    				p_tf(b, avd_length, document_length, tf))

def log_theta_p(theta, b, avd_length, document_length, tf):
    return log_concav(document_length, 
    				  theta_tf(theta, document_length, 
    				  		   p_tf(b, avd_length, document_length, tf)))

class ExtendedTfidfVectorizer:

    def __init__(self, min_df = 1):
        self.vocabulary = {}
        self.word_count = 0
        # a dictionary whose keys are terms and values are dictionaries 
        # whose keys are documents containing the terms and keys are
        # terms' frequency
        self.term_document_frequency = {}
        self.idf = {}
        self.average_length = 0
        self.num_documents = 0
        self.min_df = min_df

    def fit(self, documents):
        self.vocabulary = {}
        self.idf = {}
        self.word_count = 0

        vocabulary = {}
        self.num_documents = len(documents)
        for i in range(self.num_documents):
            doc = documents[i]
            self.average_length += len(doc.split())/self.num_documents
            unique_word = np.unique(doc.split())
            for word in unique_word:
                if word not in vocabulary:
                    vocabulary[word] = self.word_count
                    self.word_count += 1
                    #
                    self.term_document_frequency[word] = 1
                else:
                    self.term_document_frequency[word] += 1

        # set vocaublary
        for word, freq in self.term_document_frequency.items():
            if freq > self.min_df:
                self.vocabulary[word] = vocabulary[word]

        # compute idf
        for word in self.vocabulary.keys():
            num_appearances = self.term_document_frequency[word]
            val = np.log(self.num_documents/num_appearances) + 1
            if val < 1.01 :
                self.idf[self.vocabulary[word]] = 0
            else:
                self.idf[self.vocabulary[word]] = val
        # release memory
            # self.term_document_frequency = {}

    def term_frequency(self, document):
        frequency = {}
        for word in document.split():
            if word in self.vocabulary:
                if self.vocabulary[word] in frequency:
                    frequency[self.vocabulary[word]] += 1
                else:
                    frequency[self.vocabulary[word]] = 1
        frequency = np.array(sorted(frequency.items()), dtype=np.float)
        return frequency

    def transform(self, documents, tf_type='log'):
        term_document_tfidf = []
        for doc in documents:
            if doc == '':
                term_document_tfidf.append([[0,0]])
                continue

            freq = self.term_frequency(doc)
            document_length = len(doc.split())
            if tf_type == 'log':
                for i in range(len(freq)):
                    freq[i][1] = self.idf[int(freq[i][0])] * log_tf(freq[i][1])
            else:
                for i in range(len(freq)):
                    freq[i][1] = self.idf[freq[i][0]] * tf_type(document_length, 
                    											freq[i][1])
            # normalzing
            normalizing_constant = np.sqrt(np.inner(freq[:, 1], freq[:, 1]))
            freq[:, 1] = freq[:, 1]/normalizing_constant
            # print freq
            # break
            term_document_tfidf.append(freq)
        return np.array(term_document_tfidf)

    def fit_transform(self, documents, tf_type='log'):
        self.fit(documents)
        return self.transform(documents, tf_type)


