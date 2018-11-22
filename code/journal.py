import tools
from igraph import Graph
import string
import numpy as np

import os
os.chdir("D:/M1_X/X_period2/introductionToTextMining/Project_textmining/nontop_report_and_code/code")


def get_clean_journal(df):
	journals = []
	for journal in df['journal']:
		journal.replace('-', '.')
		journal.replace('..', '.')
		journal = journal.strip('.')
		punct = string.punctuation.replace('.', '')
		journal = ''.join([l for l in journal if l not in punct])
		journal = journal.lower()
		journals.append(journal)
	return journals

def unique_journals(journals):
	unique_journals = {}
	for journal in journals:
		if journal not in unique_journals:
			unique_journals[journal] = 1
	return unique_journals.keys()

def build_journal_graph(journals, citations_links, unique_journals):
	edges = {}
	for citation in citations_links:
		journal_a = journals[citation[0]]
		journal_b = journals[citation[1]]
		if journal_a != '' and journal_b != '' and citation[2] == 1:
			if (journal_a, journal_b) not in edges:
				edges[(journal_a, journal_b)] = 1
			else:
				edges[(journal_a, journal_b)] += 1

	g = Graph(directed=True)
	g.add_vertices(unique_journals)
	g.add_edges(edges.keys())
	g.es['weight'] = edges.values()
	return g

def get_journal_link_as_feature(journal_graph, journals, citations_links):
	feature = np.zeros(len(citations_links))
	for i in range(len(citations_links)):
		journal_a = journals[citations_links[i][0]]
		journal_b = journals[citations_links[i][1]]
		if journal_a != '' and journal_b != '':
			feature[i] = journal_graph[journal_a, journal_b] + \
						 journal_graph[journal_b, journal_a]
	return feature

	