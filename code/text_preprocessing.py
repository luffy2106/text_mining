import nltk
import numpy as np
from textblob import TextBlob, Word
import re
import string


import os
os.chdir("D:/M1_X/X_period2/introductionToTextMining/Project_textmining/nontop_report_and_code/code")

"""
This file contains functions to clean texts.
"""

def remove_hyphen(text):
	return text.replace('-', ' ')	

def clean_basic(text):
	# remove non ascii characters
	cleaned_text = [character for character in text 
			                  if character in string.printable]
	# remove punctuations except hyphen
	punct = string.punctuation.replace('-', '')
	cleaned_text = ''.join([l for l in cleaned_text if l not in punct])
	# lower text
	cleaned_text = cleaned_text.lower()
	return cleaned_text

def remove_short_tokens(tokens):
	return [word for word in tokens if len(word) >= 2]

def concat_short_tokens(tokens):
	new_tokens = []
	index = 0
	while index < len(tokens):
		if len(tokens[index]) == 1 and index + 1 < len(tokens):
			new_tokens.append(tokens[index] + "-" + tokens[index+1])
			index += 2
		else:
			new_tokens.append(tokens[index])
			index += 1
	return new_tokens


def remove_stopwords(tokens, stpwds):
	return [word for word in tokens if word not in stpwds]
	

def postagging(tokens, valid_tags):
	if len(tokens) == 0:
		return tokens

	blob = TextBlob(' '.join(tokens))
	new_tokens = []
	for word, tag in blob.tags:
		if (tag == 'NNS' or tag == 'NNPS') and tag in valid_tags:
			new_tokens.append(word.singularize())
		elif (tag == 'VBG' or tag == 'VBN') and tag in valid_tags:
			new_tokens.append(word.lemmatize('v')) 
		elif tag in valid_tags:
			new_tokens.append(word)
	return new_tokens

def singularize(tokens):
	return [Word(token).singularize() for token in tokens]

def stem(tokens):
	stemmer = nltk.PorterStemmer()
	new_tokens = []
	for word in tokens:
		try:
			new_tokens.append(stemmer.stem(word))
		except Exception:
			new_tokens.append(word)
	return new_tokens

def clean_text_1(text, stpwds, valid_tags, stemming=False):
	"""
	Clean_basic + postagging + remove_stopwords + remove_short_words + 
	remove_hyphen
	"""
	cleaned_text = clean_basic(text)
	tokens = cleaned_text.split()
	tokens = postagging(tokens, valid_tags)
	tokens = remove_stopwords(tokens, stpwds)
	tokens = remove_short_tokens(tokens)
	if stemming:
		tokens = stem(tokens)
	cleaned_text = remove_hyphen(' '.join(tokens))
	return cleaned_text

def clean_text_2(text, stpwds, valid_tags, stemming=False):
	"""
	Clean_basic + postagging + remove_stopwords + remove_short_words
	"""
	cleaned_text = clean_basic(text)
	tokens = cleaned_text.split()
	tokens = postagging(tokens, valid_tags)
	tokens = remove_stopwords(tokens, stpwds)
	tokens = remove_short_tokens(tokens)
	if stemming:
		tokens = stem(tokens)
	return ' '.join(tokens)
	

def clean_text_3(text, stpwds, valid_tags, stemming=False):
	"""
	Clean_basic + remove_short_words + remove_stopwords + postagging + 
	singularize
	"""
	cleaned_text = clean_basic(text)
	cleaned_text = remove_hyphen(cleaned_text)
	tokens = cleaned_text.split()
	tokens = remove_stopwords(tokens, stpwds)
	tokens = concat_short_tokens(tokens)
	tokens = remove_short_tokens(tokens)
	if stemming:
		tokens = stem(tokens)
	else:
		tokens = singularize(tokens)
	return ' '.join(tokens)

def clean_text_4(text, stpwds, stemming=False):
	"""
	Clean_basic + remove_stopwords + remove_short_words + remove_hyphen
	"""
	cleaned_text = clean_basic(text)
	tokens = cleaned_text.split()
	tokens = remove_stopwords(tokens, stpwds)
	tokens = remove_short_tokens(tokens)
	if stemming:
		tokens = stem(tokens)
	cleaned_text = remove_hyphen(' '.join(tokens))
	return cleaned_text


