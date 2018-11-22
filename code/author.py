import nltk
import tools
import numpy as np
from text_preprocessing import *
import string
from igraph import Graph

import os
os.chdir("D:/M1_X/X_period2/introductionToTextMining/Project_textmining/nontop_report_and_code/code")


"""
This file contains functions to process author column in data.
"""

def find_parenthesis(text):
    """
    Find positions a pair of parenthesis '(' and ')' in a string
    """
    begin_pos = text.find('(')
    if begin_pos == -1:
        return (-1, -1)
    else:
        enter = 1
        end_pos = -1
        for i in range(begin_pos+1, len(text)):
            if text[i] == ')':
                enter -= 1
            if text[i] == '(':
                enter += 1
            if enter == 0:
                end_pos = i
                break
        return (begin_pos, end_pos)

def remove_note(text):
    """
    Remove parts of text that are between two parenthesis.
    """
    cleaned_text = text
    while True:
        pos = find_parenthesis(cleaned_text)
        if pos[0] == -1:
            break
        elif pos[1] == -1:
            cleaned_text = cleaned_text[:pos[0]]
        else:
            cleaned_text = cleaned_text.replace(cleaned_text[pos[0]: pos[1]+1], 
                                                '')
    return cleaned_text

def clean_author_1(text):
    cleaned_text = text.lower()
    cleaned_text = remove_note(cleaned_text)
    punct = string.punctuation.replace(',','')
    punct = punct.replace('-','')
    punct = punct.replace('.','')
    cleaned_text = ''.join([l for l in cleaned_text if l not in punct])
    cleaned_text = cleaned_text.split(',')
    cleaned_text = [word.replace(' ', '.').strip('.') for word in cleaned_text]
    cleaned_text = [word for word in cleaned_text if word != '']
    return cleaned_text

def clean_author_2(text):
    cleaned_text = text.lower()
    cleaned_text = remove_note(cleaned_text)
    punct = string.punctuation.replace(',','')
    punct = punct.replace('-','')
    punct = punct.replace('.','')
    cleaned_text = ''.join([l for l in cleaned_text if l not in punct])
    cleaned_text = cleaned_text.split(',')
    cleaned_text = [word.replace(' ', '.').strip('.') for word in cleaned_text]
    cleaned_text = [word.replace('..', '.') for word in cleaned_text]
    if len(cleaned_text) == 1:
        cleaned_text = [cleaned_text[0]]
    elif len(cleaned_text) == 2:
        cleaned_text = cleaned_text[:2]
    elif len(cleaned_text) >= 3:
        cleaned_text = [cleaned_text[0], cleaned_text[1], cleaned_text[-1]]
    cleaned_text = [word for word in cleaned_text if word != '' and 
                                                     len(word) > 2]
    return cleaned_text


def get_cleaned_authors_1(df):
    """
    Clean author column in 'df'

    Parameters:

    df: a dataframe containing a columns author
    """
    authors = []
    for i in range(len(df['author'])):
        authors.append(clean_author_1(df.ix[i, 'author']))
    return authors

def get_cleaned_authors_2(df):
    """
    Clean author column in 'df'

    Parameters:

    df: a dataframe containing a columns author
    """
    authors = []
    for i in range(len(df['author'])):
        authors.append(clean_author_2(df.ix[i, 'author']))
    return authors

def unique_authors(authors):
    """
    Get a list of all different authors in the array 'authors'

    Parameters:

    authors: an array containing authors' names
    """
    list_unique_authors = {}
    for researcher_list in authors:
      if len(researcher_list) > 0:
        for researcher in researcher_list:
          if researcher in list_unique_authors:
            continue
          else:
            list_unique_authors[researcher] = 1
    return list_unique_authors.keys()

def build_graph_1(paper_graph, citation_links, 
                  authors, unique_authors):
    """
    Build a graph of authors based on paper_graph. There is an edge from
    author_a to author_b if a paper of author_a cites a paper of author_b.
    The weight of the edge (author_a, author_b) is the number of times 
    author_a cites authors_b

    Parameters:

    paper_graph: a graph of papers
    citations_links: citations information in training data. It is an array
    whose rows are triples (id of paper_a, id of paper_b, 0 or 1)
    authors: an array of authors' names
    unique_authors: an array of different authors' names
    """

    edges_between_researchers = {}
    for citation in citation_links:
        # if there is no citation link between two papers, the graph doest not 
        # change
        if citation[2] == 0:
            continue

        # in the case one of the paper in the pair cites the other
        a = citation[0]
        b = citation[1]

        # ignore the case where the list of author of a paper is not availabel
        if len(authors[a]) == 0 or len(authors[b]) == 0:
            continue
        # compute weight of links between authors
        for researcher_a in authors[a]:
            for researcher_b in authors[b]:
                # update the weight of edges
                if (researcher_a, researcher_b) in edges_between_researchers:
                    edges_between_researchers[(researcher_a, researcher_b)] += 1
                else:
                    edges_between_researchers[(researcher_a, researcher_b)] = 1

    g = Graph(directed=True)
    g.add_vertices(unique_authors)
    g.add_edges(edges_between_researchers.keys())
    g.es['weight'] = edges_between_researchers.values()

    return g

def build_graph_2(paper_graph, citation_links, 
                  authors, unique_authors):
    """
    Build a graph of authors based on paper_graph. There is an edge from
    author_a to author_b if a paper of author_a cites a paper of author_b.
    The weight of the edge (author_a, author_b) is the number of times 
    author_a cites authors_b

    Parameters:

    paper_graph: a graph of papers
    citations_links: citations information in training data. It is an array
    whose rows are triples (id of paper_a, id of paper_b, 0 or 1)
    authors: an array of authors' names
    unique_authors: an array of different authors' names
    """

    edges_between_researchers = {}
    for citation in citation_links:
        # if there is no citation link between two papers, the graph doest not 
        # change
        if citation[2] == 0:
            continue

        # in the case one of the paper in the pair cites the other
        a = citation[0]
        b = citation[1]

        # ignore the case where the list of author of a paper is not availabel
        if len(authors[a]) == 0 or len(authors[b]) == 0:
            continue

        # compute weight of links between authors
        if (authors[a][0], authors[b][0]) in edges_between_researchers:
            edges_between_researchers[(authors[a][0], authors[b][0])] += 1
        else:
            edges_between_researchers[(authors[a][0], authors[b][0])] = 1

    g = Graph(directed=True)
    g.add_vertices(unique_authors)
    g.add_edges(edges_between_researchers.keys())
    g.es['weight'] = edges_between_researchers.values()

    return g

def build_graph_3(paper_graph, citation_links, 
                  authors, unique_authors):
    
    """
    Build a graph of authors based on paper_graph. There is an edge from
    author_a to author_b if a paper of author_a cites a paper of author_b.
    The weight of the edge (author_a, author_b) is the number of times 
    author_a cites authors_b

    Parameters:

    paper_graph: a graph of papers
    citations_links: citations information in training data. It is an array
    whose rows are triples (id of paper_a, id of paper_b, 0 or 1)
    authors: an array of authors' names
    unique_authors: an array of different authors' names
    """

    edges_between_researchers = {}
    for citation in citation_links:
        # if there is no citation link between two papers, the graph doest not 
        # change
        if citation[2] == 0:
            continue

        # in the case one of the paper in the pair cites the other
        a = citation[0]
        b = citation[1]

        # ignore the case where the list of author of a paper is not availabel
        if len(authors[a]) == 0 or len(authors[b]) == 0:
            continue

        used_authors_a = []
        used_authors_b = []
        if len(authors[a]) == 1:
            used_authors_a = [authors[a][0]]
        else:
            used_authors_a = [authors[a][0], authors[a][-1]]

        if len(authors[b]) == 1:
            used_authors_b = [authors[b][0]]
        else:
            used_authors_b = [authors[b][0], authors[b][-1]]

        # compute weight of links between authors
        for researcher_a in used_authors_a:
            for researcher_b in used_authors_b:
                # update the weight of edges
                if (researcher_a, researcher_b) in edges_between_researchers:
                    edges_between_researchers[(researcher_a, researcher_b)] += 1
                else:
                    edges_between_researchers[(researcher_a, researcher_b)] = 1

    g = Graph(directed=True)
    g.add_vertices(unique_authors)
    g.add_edges(edges_between_researchers.keys())
    g.es['weight'] = edges_between_researchers.values()

    return g

def get_number_of_cited_papers(paper_graph, authors, citation_links,
                               unique_author):
    """
    For each author in authors, compute how many of his papers are cited

    Parameters:

    paper_graph: a graph of papers
    citations_links: citations information in training data. It is an array
    whose rows are triples (id of paper_a, id of paper_b, 0 or 1)
    authors: an array of authors' names
    unique_authors: an array of different authors' names
    """
    author_number_of_cited_papers = dict(zip(unique_author, 
                                             np.zeros(len(unique_author))))
    paper_number_of_citations = paper_graph.indegree()
    for citation in citation_links:
        for researcher in authors[citation[1]]:
            author_number_of_cited_papers[researcher] += \
                                        paper_number_of_citations[citation[1]]
    return author_number_of_cited_papers

def get_number_of_cited_papers_as_feature(paper_graph, authors, 
                                                 citation_links, unique_author,
                                                 criterion='max'):
    """
    For each row (paper_a, paper_b, is_linked) in citation_links, compute
    the total numbers of citations of authors of paper_b

    Parameters:

    paper_graph: a graph of papers
    citations_links: citations information in training data. It is an array
    whose rows are triples (id of paper_a, id of paper_b, 0 or 1)
    authors: an array of authors' names
    unique_authors: an array of different authors' names
    """
    
    author_number_of_cited_papers = get_number_of_cited_papers(paper_graph, 
                                        authors, citation_links, unique_author)
    feature = np.zeros(len(citation_links))
    for i in range(len(citation_links)):
        if len(authors[citation_links[i][1]]) > 0:
            if criterion == 'mean':
                feature[i] = np.mean([author_number_of_cited_papers[researcher]
                           for researcher in authors[citation_links[i][1]]])
            elif criterion == 'sum':
                feature[i] = np.sum([author_number_of_cited_papers[researcher]
                             for researcher in authors[citation_links[i][1]]])
            else:
                feature[i] = np.max([author_number_of_cited_papers[researcher]
                             for researcher in authors[citation_links[i][1]]])
    return feature

# def get_author_degree(author_graph, )

def get_researcher_double_link_as_feature(author_graph, authors, citation_links,
                                          criterion='mean'):
    """
    For each pair (paper_a, paper_b), compute the number of links between authors 
    of the two papers. Links are counted in two directions.
    """
    feature = np.zeros(len(citation_links))
    for i in range(len(citation_links)):
        a = citation_links[i][0]
        b = citation_links[i][1]
        
        if len(authors[a]) == 0 or len(authors[b]) == 0:
            continue

        if criterion == 'sum':
            feature[i] = np.sum([author_graph[author_a, author_b] + \
                                 author_graph[author_b, author_a]
                                 for author_a in authors[a]
                                 for author_b in authors[b]])
        elif criterion == 'max':
            feature[i] = np.max([author_graph[author_a, author_b] + \
                                  author_graph[author_b, author_a]
                                  for author_a in authors[a]
                                  for author_b in authors[b]])
        else:
            feature[i] = np.mean([author_graph[author_a, author_b] + \
                                 author_graph[author_b, author_a]
                                 for author_a in authors[a]
                                 for author_b in authors[b]])
    return feature


def get_researcher_link_as_feature(author_graph, authors, 
                                            citation_links, criterion='mean'):
    """
    For each pair (paper_a, paper_b), compute the number of links between authors 
    of the two papers. Links are directed.
    """
    feature = np.zeros(len(citation_links))
    for i in range(len(citation_links)):
        a = citation_links[i][0]
        b = citation_links[i][1]
        
        if len(authors[a]) == 0 or len(authors[b]) == 0:
            continue

        if criterion == 'sum':
            feature[i] = np.sum([author_graph[(author_a, author_b)]
                                 for author_a in authors[a]
                                 for author_b in authors[b]])
        elif criterion == 'max':
            feature[i] = np.max([author_graph[(author_a, author_b)]
                                 for author_a in authors[a]
                                 for author_b in authors[b]])
        else:
            feature[i] = np.mean([author_graph[(author_a, author_b)]
                                 for author_a in authors[a]
                                 for author_b in authors[b]])
    return feature

def get_researcher_link_as_feature_2(author_graph, authors, 
                                            citation_links, criterion='mean'):
    """
    For each pair (paper_a, paper_b), compute the number of links between authors 
    of the two papers. Links are directed.
    """
    feature = np.zeros(len(citation_links))
    for i in range(len(citation_links)):
        a = citation_links[i][0]
        b = citation_links[i][1]

        
        
        if len(authors[a]) == 0 or len(authors[b]) == 0:
            continue

        used_authors_b = [authors[b][0]]

        if criterion == 'sum':
            feature[i] = np.sum([author_graph[(author_a, author_b)]
                                 for author_a in authors[a]
                                 for author_b in used_authors_b])
        elif criterion == 'max':
            feature[i] = np.max([author_graph[(author_a, author_b)]
                                 for author_a in authors[a]
                                 for author_b in used_authors_b])
        else:
            feature[i] = np.mean([author_graph[(author_a, author_b)]
                                 for author_a in authors[a]
                                 for author_b in used_authors_b])
    return feature

def build_coauthor_graph_1(citation_links, authors, unique_authors):
    """
    Build a graph whose nodes are authors. There is an edge between 
    two authors if they work together in a papers.
    """
    coauthors = {}
    for author_list in authors:
        for i in range(len(author_list)):
            for j in range(i):
                if (author_list[i], author_list[j]) in coauthors  \
                    or (author_list[j], author_list[i]) in coauthors:
                    coauthors[(author_list[i], author_list[j])] += 1
                    coauthors[(author_list[j], author_list[i])] += 1
                else:
                    coauthors[(author_list[i], author_list[j])] = 1
                    coauthors[(author_list[j], author_list[i])] = 1

    g = Graph(directed=False)
    g.add_vertices(unique_authors)
    g.add_edges(coauthors.keys())
    g.es['weight'] = coauthors.values()

    return g

def get_coauthor_link_as_feature(coauthor_graph, authors, 
                                 citation_links, criterion='max'):
    feature = np.zeros(len(citation_links))
    for i in range(len(citation_links)):
        a = citation_links[i][0]
        b = citation_links[i][1]
        for author_a in authors[a]:
            for author_b in authors[b]:
                if criterion == 'sum':
                    feature[i] += coauthor_graph[(author_a, author_b)]
                else:
                    feature[i] = max(feature[i], 
                                 coauthor_graph[(author_a, author_b)])
    return feature

def get_common_neighbor_as_feature(author_graph, authors, citation_links, mode='all'):
#    from sets import Set
    common_neighbor = np.empty(len(citation_links))
    for i in range(len(citation_links)):
        if len(authors[citation_links[i][0]]) == 0 or \
            len(authors[citation_links[i][1]]) == 0:
            common_neighbor[i] = 0
        else:
            neighborhood_a = neighborhood_b = set([])
            for author_a in authors[citation_links[i][0]]:
                neighborhood_a = neighborhood_a.union(set(author_graph.neighbors(
                                                      author_a, mode=mode)))
            for author_b in authors[citation_links[i][1]]:
                neighborhood_b = neighborhood_b.union(set(author_graph.neighbors(
                                                      author_b, mode=mode)))
            common_neighbor[i] = len(neighborhood_a.intersection(
                                                     neighborhood_b))
    return common_neighbor