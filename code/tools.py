"""
This file contains functions to read/write data and some basic functions to
process data.
This script can be executed in Python 2.* only.
"""
from __future__ import division
import pandas as pd
import csv
import re
import igraph
import numpy as np
import nltk
import gensim
from sklearn.metrics import f1_score, recall_score, precision_score


import os
os.chdir("D:/M1_X/X_period2/introductionToTextMining/Project_textmining/nontop_report_and_code/code")

def normalize_matrix(data, means, is_normalized=None):
    """
    Normalize each column of data to std=1 and mean specified by 'means'

    Parameters:

    data: type = numpy array
    means: an array whose length is equal to number of columns in data, type = 
    numpy array
    is_normalized: an boolean array indicating which columns need to be 
    normalized, type = numpy array
    """
    new_data = np.empty(np.array(data).shape)
    column_means = np.mean(data, axis=0)
    column_stds = np.std(data, axis=0)
    for i in range(len(column_means)):
        if is_normalized is None or is_normalized[i] == True:
            new_data[:, i] = (data[:, i] - column_means[i] + \
                             means[i])/column_stds[i]
        else:
            new_data[:, i] = data[:, i]
    return new_data

def normalized_feature(feature, mean = 0.0):
    """
    Normalize a feature to an std = 1 and mean specified by parameter 'mean'

    Parameters:

    features: numpy array
    mean: scalar
    """
    feature_mean = np.mean(feature)
    feature_std = np.std(feature)
    feature_ = np.array(feature, copy=True)
    return (feature_ - feature_mean)/feature_std



def id_to_time(paper_id):
    """
    Get year and month of publication of papers from their id

    Parameters:

    paper_id: id of a paper, type = integer

    Results: A pair (year, month)
    """
    if len(str(paper_id)) <= 5:
        return (2000, paper_id/1000)
    elif len(str(paper_id)) == 6:
        return (2000 + paper_id/100000, 
                (paper_id - paper_id/100000*100000)/1000)
    else:
        return (1900 + paper_id/100000, 
                (paper_id - paper_id/100000*100000)/1000)

def get_submission_time(df):
    """
    Get year and month of publication of all papers in dataframe

    Parameters:
    df: Dataframe read from read_node, type = pandas Dataframe
    """
    submission_time = []
    for i in range(len(df['id'])):
        submission_time.append(id_to_time(df['id'][i]))
    return np.array(submission_time)

def compare_time(time_a, time_b):
    """
    Compare publication time of two papers

    Parameters:
    time_a: a pair (year, month)
    time_b: a pair (year_month)
    """
    diff = np.empty(len(time_a))
    for i in range(len(time_a)):
        if time_a[i][0] != time_b[i][0]:
            diff[i] = np.sign(time_a[i][0] - time_b[i][0])
        else:
            diff[i] = np.sign(time_a[i][1] - time_b[i][1])
    return diff

def read_citations(dict_node):
    """
    Read training data into a numpy array. Each in the training data
    corresponds to a row of the numpy array. A row of the numpy array
    is an array of 3 elements: the first and the second element are 
    indices of papers while the third is 0 or 1 indicating whether
    there is a citation link between the two papers

    Parameters:
    dict_node: dictionary built by function build_dict_node, type = dictionary
    """
    citations = []
    with open("training_set.txt", "r") as file:
        for i, line in enumerate(file):
            citation = re.split(' |\n|\r', line)[:3]
            # convert papers' ids to indices
            citation = [int(el) for el in citation]
            citation[0] = dict_node[citation[0]]
            citation[1] = dict_node[citation[1]]
            # add to the list
            citations.append(citation)
    return np.array(citations, dtype=int)

def build_dict_node(df):
    """
    Build a dictionary mapping each paper id to an index in 
    [0..num_of_paper - 1]
    """
    dict_node = dict()
    for i in range(len(df['id'])):
        dict_node[df['id'][i]] = i
    return dict_node

def read_node():
    """
    Read data from node_information.csv to a dataframe
    """
    df = pd.read_csv("node_information.csv", header=None)
    df.columns = ['id', 'year', 'title', 'author', 'journal', 'abstract']
    # transform column 'author'
    df.ix[df['author'].isnull(), 'author'] = ''
    # transform column 'journal'
    df.ix[df['journal'].isnull(), 'journal'] = ''
    return df

def read_test(dict_node):
    test = []
    with open("testing_set.txt", 'r') as file:
        for line in file:
            test.append([dict_node[int(node)] for node in line.split()])
    return np.asarray(test)

def test_feature(training_data, validation_data, y_train, y_validation, 
                 model=None):
    train = training_data.reshape(-1, 1)
    validation = validation_data.reshape(-1, 1)
    if model is None:
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression(C=0.3)
    model_info = model.fit(train, y_train)
    test_model(model, train, validation, y_train, y_validation)

def test_probabilistic_model(model, train, validation, y_train, y_validation, 
                             threshold=0.5, fitted=False, verbose=True):
    """
    Test performance of a probabilistic classifier model (which implements 
    functions telling probability of classes).

    Parameters:

    model: a sklearn classifier
    train: training data, type = np.array
    validation: validation data, type = np.array
    y_train: labels in training data, type = np.array
    y_validation labels in validation data, type = np.array
    threshold: a threshold for probability above which we assign the label of 
    a row to 1, type = scalar
    fitted: informs if 'model' is already fitted, type = boolean
    """
    if not fitted:
        model_info = model.fit(train, y_train)
    proba_train = model.predict_proba(train)
    proba_validation = model.predict_proba(validation)
    pred_train = np.array(proba_train[:, 1] >= threshold, dtype=int)
    pred_validation = np.array(proba_validation[:, 1] >= threshold, dtype=int)
    if verbose:
        print("Precision, recall and f1_score on train data: ")
        print(precision_score(y_train, pred_train), \
              recall_score(y_train, pred_train), f1_score(y_train, pred_train))
        print("F1 score on validation data: ")
        print(precision_score(y_validation, pred_validation), \
              recall_score(y_validation, pred_validation), \
              f1_score(y_validation, pred_validation))
    return f1_score(y_validation, pred_validation)

def test_model(model, train, validation, y_train, y_validation, fitted=False, 
               verbose=True):
    """
    Test performace of a model by computing f1_score

    Parameters:
    
    model: a classifier in sklearn
    train: training data
    validation: validation data
    y_train: labels in training data
    y_validation: labels in validation data
    """
    if not fitted:
        model_info = model.fit(train, y_train)
    pred_train = model.predict(train)
    pred_validation = model.predict(validation)
    if verbose:
        print("Precision, recall and f1_score on train data: ")
        print(precision_score(y_train, pred_train), \
              recall_score(y_train, pred_train), f1_score(y_train, pred_train))
        print("F1 score on validation data: ")
        print(precision_score(y_validation, pred_validation), \
              recall_score(y_validation, pred_validation), \
              f1_score(y_validation, pred_validation))
    return f1_score(y_validation, pred_validation)

def test_adaboost(learners, train, validation, y_train, y_validation, 
                  num_iteration=20):
    assert(len(learners) >= train.shape[1])
    for i in range(len(learners)):
        learners[i].fit(train[:, i].reshape(-1, 1), y_train)
    import adaboost
    weak_predictions_train = np.array(
                            [learners[i].predict(train[:, i].reshape(-1,1)) 
                             for i in range(len(learners))])
    weak_predictions_validation = np.array(
                            [learners[i].predict(validation[:, i].reshape(-1,1)) 
                             for i in range(len(learners))])
    y_train_new = np.array((y_train - 0.5) * 2, int)
    for prediction in weak_predictions_train:
        prediction[prediction == 0] = -1
    for prediction in weak_predictions_validation:
        prediction[prediction == 0] = -1
    boosting_coef = adaboost.boost(len(learners), weak_predictions_train, 
                                   num_iteration, y_train_new)
    pred_validation = np.array([0] * len(y_validation), float)
    f1_scores = []
    recalls = []
    precisions = []
    feature_importance = [0.0] * len(learners)
    for i in range(num_iteration):
        feature_importance[boosting_coef[i][0]] += boosting_coef[i][1]
    for i in range(num_iteration):
        pred_validation += boosting_coef[i][1] * \
                           weak_predictions_validation[boosting_coef[i][0], :]
        pred = np.array(pred_validation >= 0, int)
        f1_scores.append(f1_score(y_validation, pred))
        recalls.append(recall_score(y_validation, pred))
        precisions.append(precision_score(y_validation, pred))
    return (precisions, recalls, f1_scores), feature_importance

def write_result(filename, result):
    """
    Write predictions to a file

    Parameters:

    filename: name of a file to write predictions
    result: an array containing predictions (0 or 1), type = numpy array
    """
    with open(filename, 'w') as file:
        writer = csv.writer(file, delimiter=',')
        writer.writerow(['id', 'category'])
        for i in range(len(result)):
            writer.writerow([i, result[i]])

def read_paper_distance():
    with open('paper_distance.txt', 'r') as f:
        paper_distance = {}
        for line in f:
            data = line.split()
            paper_distance[(int(data[0]), int(data[1]))] = float(data[2])
        return paper_distance

def read_similarity():
    sim = []
    with open("abstract-similarities.txt", 'r') as f:
        while True:
            try:
                sim.append(float(f.readline()))
            except Exception:
                break
    return np.asarray(sim)


def get_keywords(g):
    score = dict(zip(g.vs["name"],g.coreness()))
    max_score = np.max(score.values())
    res = [word for word in score.keys() if score[word] == max_score]
    return res



