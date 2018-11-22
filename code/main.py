from __future__ import division
import sys
import tools
import abstract
import title
import paper
import author
import text_preprocessing
import my_tfidf
import journal
import nltk
from textblob import *
import gensim
import numpy as np
from sklearn.linear_model import LogisticRegression, SGDClassifier, LogisticRegressionCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score, recall_score, precision_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import VotingClassifier
from sklearn.decomposition import PCA
from imp import reload as rl
from collections import OrderedDict


import os
os.chdir("D:/M1_X/X_period2/introductionToTextMining/Project_textmining/nontop_report_and_code/code")

#if len(sys.argv) == 1:
#	print('Type file name to store result...')
#	exit(0)
#else:
#	file_name = sys.argv[1]

file_name =  "result.txt"

nltk.download('stopwords')
stpwds = set(nltk.corpus.stopwords.words("english"))

valid_tag = ['NN', 'NNS', 'NNP', 'NNPS', 'JJ', 'JJS', 'JJP', 'VBG', 'VBN']

print("Reading input")
df = tools.read_node()
dict_node = tools.build_dict_node(df)
citations_train = tools.read_citations(dict_node)
citations_test = tools.read_test(dict_node)
y = citations_train[:, 2]

print("Cleaning abstracts and titles")
num_papers = len(df['title'])
abstracts = abstract.clean_abstract_1(df, stpwds, valid_tag, stemming=False, verbose=True)
titles = title.clean_title_1(df, stpwds, valid_tag, stemming=False, verbose=True)
title_abstract = [title_ + ' ' + abstract_ for title_, abstract_ in zip(titles, abstracts)]

print("Computing content similarity using tf-idf")
my_vectorizer = my_tfidf.ExtendedTfidfVectorizer(min_df=1)
tfidf_matrix = my_vectorizer.fit_transform(title_abstract)

tfidf_title_plus_abstract_train = np.empty(len(citations_train))
for i in range(len(citations_train)):
    if i % 100000 == 0:
        print("Processing the %dth row in train..."%i)
    tfidf_title_plus_abstract_train[i] = my_tfidf.mul_sparse_vector(tfidf_matrix[citations_train[i][0]],                                                          
                                 tfidf_matrix[citations_train[i][1]])

tfidf_title_plus_abstract_test = np.empty(len(citations_test))
for i in range(len(citations_test)):
    if i % 10000 == 0:
        print("Processing the %dth row in test..."%i)
    tfidf_title_plus_abstract_test[i] = my_tfidf.mul_sparse_vector(tfidf_matrix[citations_test[i][0]],                                                          
                                 tfidf_matrix[citations_test[i][1]])


#Second run

#Note : we use log function to reduce skew
# paper graph (the intuition in here is if the paper is popular, it will be highly sited from another paper
paper_graph = paper.build_graph_1(citations_train, dict_node)
paper_degree_train = paper.get_degree_as_feature(paper_graph, citations_train)  #each paper will have degree
paper_degree_log_train = np.log(1 + paper_degree_train)
paper_degree_test = paper.get_degree_as_feature(paper_graph, citations_test)
paper_degree_log_test = np.log(1 + paper_degree_test)


#multiple degree of a pair of paper, from this we will have the number of paper in a connected graph which is created by this pair of paper
print("Computing preferential attachment")
pa_train = np.multiply(np.array(paper_graph.degree())[citations_train[:, 1]], np.array(paper_graph.degree())[citations_train[:, 0]])
pa_log_train = np.log(1 + pa_train)
pa_test = np.multiply(np.array(paper_graph.degree())[citations_test[:, 1]], np.array(paper_graph.degree())[citations_test[:, 0]])
pa_log_test = np.log(1 + pa_test)

# common neighbors, aa, ra
common_neighbor_train, aa_train, ra_train = paper.get_common_neighbor_aa_ra(paper_graph, citations_train)
common_neighbor_log_train = np.log(1 + common_neighbor_train)
common_neighbor_test, aa_test, ra_test = paper.get_common_neighbor_aa_ra(paper_graph, citations_test)
common_neighbor_log_test = np.log(1 + common_neighbor_test)

# author graph (we do the same things as the way we build a graph of author)
# #ra_train = 615512, #aa_train = 615512, common_neighbor_train = 615512
authors = author.get_cleaned_authors_2(df)
#unique_authors = author.unique_authors(authors) #error in here
# replace by Kien but still error(fix later)
unique_authors = list(OrderedDict.fromkeys(authors))
author_graph = author.build_graph_1(paper_graph, citations_train, authors, unique_authors)
author_link_train = author.get_researcher_link_as_feature(author_graph, authors, citations_train, 'sum')
author_link_test = author.get_researcher_link_as_feature(author_graph, authors, citations_test, 'sum')

print("Computing common neighbor authors...")
common_neighbor_author_train = author.get_common_neighbor_as_feature(author_graph, authors, citations_train, 'all')
common_neighbor_author_train = np.log(1 + common_neighbor_author_train)
common_neighbor_author_test = author.get_common_neighbor_as_feature(author_graph, authors, citations_test, 'all')
common_neighbor_author_test = np.log(1 + common_neighbor_author_test)

# empty author (to check if one of the two paper has no author)
is_empty_author_train = np.array(np.logical_and(np.asarray(df['author'])[citations_train[:, 0]] == '',
                                      np.asarray(df['author'])[citations_train[:, 1]] == ''),
                        dtype=int)
is_empty_author_test = np.array(np.logical_and(np.asarray(df['author'])[citations_test[:, 0]] == '',
                                      np.asarray(df['author'])[citations_test[:, 1]] == ''),
                        dtype=int)

# submission time: The intituition here is if the duration of submisson is near each other, it is highly possible that they will be cited from each other
submission_time = tools.get_submission_time(df)
submission_time_diff_train = tools.compare_time(submission_time[citations_train[:, 0]], submission_time[citations_train[:, 1]])
submission_time_diff_train[submission_time_diff_train == 0] = 1
submission_time_diff_test = tools.compare_time(submission_time[citations_test[:, 0]], submission_time[citations_test[:, 1]])
submission_time_diff_test[submission_time_diff_test == 0] = 1


# similarity jaccard
similarity_jaccard_train = paper.get_similarity_jaccard_as_feature(paper_graph, citations_train, 'all')
similarity_jaccard_test = paper.get_similarity_jaccard_as_feature(paper_graph, citations_test, 'all')

# similarity dice
similarity_dice_train = paper.get_similarity_dice_as_feature(paper_graph, citations_train, 'all')
similarity_dice_test = paper.get_similarity_dice_as_feature(paper_graph, citations_test, 'all')



# create training data and test data (here will be the column in our data set)
features = ['tfidf_title_plus_abstract', 
			'common_neighbor_log',
			'aa',
			'pa_log', 
			'paper_degree_log', 
			'similarity_dice', 
			'similarity_jaccard',
			'submission_time_diff',
			'author_link',
			'common_neighbor_author',
			'is_empty_author'
			]
train_data = np.column_stack([eval(feature + '_train') for feature in features])
test_data = np.column_stack([eval(feature + '_test') for feature in features])

model = LogisticRegression(C=0.3) #we choose logistic regression since it show the best result
model_info = model.fit(train_data, y)
pred_test_proba = model.predict_proba(test_data)
pred_test = np.array(pred_test_proba[:, 1] >= 0.35, dtype=int)
tools.write_result(file_name, pred_test)






