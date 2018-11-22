from __future__ import division
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
import time

import os
os.chdir("D:/M1_X/X_period2/introductionToTextMining/Project_textmining/nontop_report_and_code/code")

# year_diff_sign helps
nltk.download('stopwords')
stpwds = set(nltk.corpus.stopwords.words("english"))

valid_tag = ['NN', 'NNS', 'NNP', 'NNPS', 'JJ', 'JJS', 'JJP', 'VBN', 'VBG']

# read input data

df = tools.read_node()
dict_node = tools.build_dict_node(df)
citations = tools.read_citations(dict_node)

# suffle rows in citations

idx = np.random.permutation(len(citations))
citations = citations[idx]
y = citations[:, 2]

num_papers = len(df['title'])
abstracts = abstract.clean_abstract_1(df, stpwds, valid_tag, stemming=False, verbose=True)
titles = title.clean_title_1(df, stpwds, valid_tag, stemming=False, verbose=True)
title_abstract = [title_ + ' ' + abstract_ for title_, abstract_ in zip(titles, abstracts)]

submission_time = tools.get_submission_time(df)
submission_time_diff = tools.compare_time(submission_time[citations[:, 0]], submission_time[citations[:, 1]])

authors = author.get_cleaned_authors_2(df)
unique_authors = author.unique_authors(authors)
journals = journal.get_clean_journal(df)
unique_journals = journal.unique_journals(journals)

is_empty_author = np.array(np.logical_and(np.asarray(df['author'])[citations[:, 0]] == '', 
                                          np.asarray(df['author'])[citations[:, 1]] == ''), 
                            dtype=int)
is_empty_journal = np.array(np.asarray(df['journal'])[citations[:, 1]] == '', dtype=int)

my_vectorizer = my_tfidf.ExtendedTfidfVectorizer(min_df=1)
tfidf_matrix = my_vectorizer.fit_transform(title_abstract)
# tfidf_matrix_title = my_vectorizer.transform(titles)
#
tfidf_title_plus_abstract = np.empty(len(citations))
for i in range(len(citations)):
    if i % 100000 == 0:
        print("Processing the %dth row in valid train..."%i)
    tfidf_title_plus_abstract[i] = my_tfidf.mul_sparse_vector(tfidf_matrix[citations[i][0]],                                                          
                                 tfidf_matrix[citations[i][1]])

V = 10
n = len(citations)
result_logistic = {}
result_xgboost = {}
result_random_forest = {}
result_random_forest['entropy'] = []
result_random_forest['gini'] = []
result_linearSVC = {}
result_decision_tree = {}
result_decision_tree[(12, 50)] = []
result_sgd = {}
result_sgd[('huber', 0.15)] = []
result_sgd[('huber', 0)] = []
for c in [1,3,0.1,0.3]:
    result_xgboost[c] = []
    result_linearSVC[c] = []
    for threshold in [0.3, 0.35, 0.4, 0.45]:
        result_logistic[(c, threshold)] = []


for i in range(10):
    print("\n\n\n")
    print("Iteration %d"%(i+1))
    # divide data
    #
    index_train = range(i*n//V) + range((i+1)*n//V, n)
    index_validation = range(i*n//V, (i+1)*n//V)
    #
    citations_train = citations[index_train]
    citations_validation = citations[index_validation]
    #
    y_train = citations[index_train, 2]
    y_validation = citations[index_validation, 2]
    #
    #
    tfidf_title_plus_abstract_train = tfidf_title_plus_abstract[index_train]
    tfidf_title_plus_abstract_validation = tfidf_title_plus_abstract[index_validation]
    #
    paper_graph = paper.build_graph_1(citations_train, dict_node)
    paper_degree = paper.get_degree_as_feature(paper_graph, citations)
    #
    print("Computing paper degree...")
    paper_degree_log = np.log(paper_degree + 1)
    paper_degree_train = paper_degree[index_train]
    paper_degree_validation = paper_degree[index_validation]
    paper_degree_log_train = paper_degree_log[index_train]
    paper_degree_log_validation = paper_degree_log[index_validation]
    #
    print("Computing preferential attachment")
    pa = np.multiply(np.array(paper_graph.degree())[citations[:, 1]], np.array(paper_graph.degree())[citations[:, 0]])
    pa_log = np.log(1 + pa)
    pa_log_train = pa_log[index_train]
    pa_log_validation = pa_log[index_validation]
    #
    print("Computing paper common neighbors...")
    common_neighbor, aa, ra = paper.get_common_neighbor_aa_ra(paper_graph, citations)
    common_neighbor_train = common_neighbor[index_train]
    common_neighbor_validation = common_neighbor[index_validation]
    common_neighbor_log = np.log(1 + common_neighbor)
    common_neighbor_log_train = common_neighbor_log[index_train]
    common_neighbor_log_validation = common_neighbor_log[index_validation]
    aa_train, aa_validation = aa[index_train], aa[index_validation]
    ra_train, ra_validation = ra[index_train], ra[index_validation]
    aa_log, ra_log = np.log(1 + aa), np.log(1 + ra)
    aa_log_train, aa_log_validation = aa_log[index_train], aa_log[index_validation]
    ra_log_train, ra_log_validation = ra_log[index_train], ra_log[index_validation]
    # similarity jaccard
    print("Computing paper similarity jaccard...")
    similarity_jaccard = paper.get_similarity_jaccard_as_feature(paper_graph, citations, 'all')
    similarity_jaccard_train = similarity_jaccard[index_train]
    similarity_jaccard_validation = similarity_jaccard[index_validation]
    # similarity dice
    print("Computing paper similarity dice...")
    similarity_dice = paper.get_similarity_dice_as_feature(paper_graph, citations, 'all')
    similarity_dice_train = similarity_dice[index_train]
    similarity_dice_validation = similarity_dice[index_validation]
    #
    #
    author_graph = author.build_graph_1(paper_graph, citations_train, authors, unique_authors)
    author_link = author.get_researcher_link_as_feature(author_graph, authors, citations, 'sum')
    #
    # author link
    #
    author_link_train = author_link[index_train]
    author_link_validation = author_link[index_validation]
    #
    print("Computing common neighbor authors...")
    common_neighbor_author = author.get_common_neighbor_as_feature(author_graph, authors, citations, 'all')
    common_neighbor_author = np.log(1 + common_neighbor_author)
    common_neighbor_author_train = common_neighbor_author[index_train]
    common_neighbor_author_validation = common_neighbor_author[index_validation] 
    #
    journal_graph = journal.build_journal_graph(journals, citations_train, unique_journals)
    journal_link = journal.get_journal_link_as_feature(journal_graph, journals, citations)
    journal_link = np.log(1 + np.log(1 + journal_link))
    journal_link_train = journal_link[index_train]
    journal_link_validation = journal_link[index_validation]
    #
    submission_time_diff_train = submission_time_diff[index_train]
    submission_time_diff_train[submission_time_diff_train == 0] = 1
    submission_time_diff_validation = submission_time_diff[index_validation]
    submission_time_diff_validation[submission_time_diff_validation == 0] = 1
    #
    is_empty_author_train = is_empty_author[index_train]
    is_empty_author_validation = is_empty_author[index_validation]
    #
    is_empty_journal_train = is_empty_journal[index_train]
    is_empty_journal_validation = is_empty_journal[index_validation]
    #
    print("Creating training and validation data...")
    features = np.array(['tfidf_title_plus_abstract', #0
                         'paper_degree_log', #3
                         'common_neighbor_log', #4
                         'pa_log',
                         'aa',
                         'submission_time_diff',
                         'author_link',
                         'common_neighbor_author',
                         'is_empty_author',
                         'similarity_dice',
                         'similarity_jaccard'
                         ])
    #
    train = np.column_stack([eval(feature + '_train') for feature in features])
    validation = np.column_stack([eval(feature + '_validation') for feature in features])
    #
    import xgboost as xgb
    for C in [1,3,0.1,0.3]:
        print("Logistic regression C=%f..."%C)
        now = time.time()
        model = LogisticRegression(C=C)
        model_info = model.fit(train, y_train)
        print("Time logistic %f"%(time.time() - now))
        now = time.time()
        result_xgboost[C].append(tools.test_model(xgb.XGBClassifier(max_depth=3, n_estimators=30, nthread=4, learning_rate=0.1, reg_alpha=C/3), 
                                                  train, validation, y_train, y_validation, verbose=False))
        print("Time xgboost %f"%(time.time() - now))
        now = time.time()
        result_linearSVC[C].append(tools.test_model(LinearSVC(C = C), train, validation, y_train, y_validation, verbose=False))
        print("Time linearSVC %f"%(time.time() - now))
        for threshold in [0.3, 0.35, 0.4, 0.45]:
            print("threshold = %f ..."%threshold)
            result_logistic[(C, threshold)].append(tools.test_probabilistic_model(model, train, validation, y_train, y_validation, threshold, fitted=True, verbose=False))
            # tools.test_probalistic_model(model, train, validation, y_train, y_validation, threshold, True)
        print("")
    #
    print("SGDClassifier(modified_huber, elasticnet)...")
    now = time.time()
    model = SGDClassifier(loss='modified_huber', penalty='elasticnet', l1_ratio=0.15)
    result_sgd[('huber', 0.15)].append(tools.test_model(model, train, validation, y_train, y_validation, verbose=False))
    print("Time sgd %f"%(time.time() - now))
    print("SGDClassifier(modified_huber, l2)...")
    now = time.time()
    model = SGDClassifier(loss='modified_huber', penalty='l2')
    result_sgd[('huber', 0)].append(tools.test_model(model, train, validation, y_train, y_validation, verbose=False))
    print("Time sgd %f"%(time.time() - now))
    print("Random forest(entropy, max_depth=12, leaf=50)... ")
    now = time.time()
    model = RandomForestClassifier(n_estimators=20,criterion='entropy', min_samples_leaf=50, max_depth=12)
    result_random_forest['entropy'].append(tools.test_model(model, train, validation, y_train, y_validation, verbose=False))
    print("Time random forest %f"%(time.time() - now))
    print("Random forest(gini, max_depth=12, leaf=50)... ")
    now = time.time()
    model = RandomForestClassifier(n_estimators=20,criterion='gini', min_samples_leaf=50, max_depth=12)
    result_random_forest['gini'].append(tools.test_model(model, train, validation, y_train, y_validation, verbose=False))
    print("Time random forest %f"%(time.time() - now))
    print("Decision Tree(max_depth=12, impurity=0.001, min_sample_split=50")
    now = time.time()
    model = DecisionTreeClassifier(min_samples_leaf=50, max_depth=12, min_impurity_split=0.001)
    result_decision_tree[(12, 50)].append(tools.test_model(model, train, validation, y_train, y_validation, verbose=False))
    print("Time decision tree %f"%(time.time() - now))

