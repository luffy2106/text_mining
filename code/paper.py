from __future__ import division
import nltk
import tools
import numpy as np
import igraph


import os
os.chdir("D:/M1_X/X_period2/introductionToTextMining/Project_textmining/nontop_report_and_code/code")

"""
This script contains function to build graphs of papers and extract features
from the graphs.
"""


def build_graph_1(citation_links, dict_node):
    """
    Build a directed graph of papers. There is an edge from paper_a to paper_b
    if paper_a cites paper_b.
    """

    edges = []
    for citation in citation_links:
        if citation[2] == 1:
	        edges.append((citation[0], citation[1]))

    g = igraph.Graph(directed=True)
    g.add_vertices(range(len(dict_node.keys())))
    g.add_edges(edges)
    return g

def compute_degree_of_all_papers(paper_graph):
	"""
	Compute the indegree of each paper in the paper_graph
	"""
	# the graph needs to be directed
	assert paper_graph.is_directed()
	# compute weighted degree for each vertex
	return np.array(paper_graph.indegree())

def get_degree_as_feature(paper_graph, citation_links):
	degrees = compute_degree_of_all_papers(paper_graph)
	return degrees[citation_links[:, 1]]

def get_pagerank_as_feature(paper_graph, citation_links):
	pagerank = np.array(paper_graph.personalized_pagerank())
	return pagerank[citation_links[:, 1]]
def get_paper_distance_as_feature(paper_graph, citation_links, verbose=False):
	"""
	For each pair (paper_a, paper_b), compute the length of the shortest path
	in the directed graph 'paper_graph' from paper_a to paper_b
	"""
	# for each source vertex in citation_links, find all of its correspondent
	# target vertex in citation_links
#	from sets import Set
	if verbose:
		print("Building a dictionary source-target...")
	source_target = {}
	for citation in citation_links:
		if citation[0] in source_target:
			source_target[citation[0]].append(citation[1])
		else:
			source_target[citation[0]] = [citation[1]]

	if verbose:
		print("Computing shortest path...")
	count = 0
	paper_distance = {}
	for source in source_target.keys():
		neighbors_set = list(set(source_target[source]).intersection(
						set(paper_graph.neighbors(source, 'out'))))
		non_neighbors_set = list(set(source_target[source]).difference(
						set(paper_graph.neighbors(source, 'out'))))
		# for target in neighbors set
		for target in neighbors_set:
			paper_graph.delete_edges([(source, target)])
			paper_distance[(source, target)] = \
								paper_graph.shortest_paths_dijkstra(source,
				   				target)[0][0]
			paper_graph.add_edges([(source, target)])
		# for targets which are not a neighbor of source
		non_neighbors_distance = paper_graph.shortest_paths_dijkstra(source,
								 non_neighbors_set)[0]
		index = 0
		for target in non_neighbors_set:
			paper_distance[(source, target)] = non_neighbors_distance[index]
			index += 1

		count += 1
		if count % 1000 == 0 and verbose:
			print("Processing the %dth source"%count)

	feature = np.empty(len(citation_links))
	for i in range(len(citation_links)):
		feature[i] = paper_distance[(citation_links[i][0], 
									 citation_links[i][1])]
	return feature 

def get_valid_common_neighbor_as_feature(paper_graph, citation_links):
	"""
	For each pair of papers (paper_a, paper_b), compute the number of valid 
	common neighbors between paper_a and paper_b. A common neighbor is valid if
	it is not simultaneously in the in-neighbors of paper_a and out-neighbors of
	paper_b
	"""
#	from sets import Set
	common_neighbor = np.empty(len(citation_links))
	for i in range(len(citation_links)):
		neighborhood_a = paper_graph.neighbors(citation_links[i][0], mode='all')
		neighborhood_b = paper_graph.neighbors(citation_links[i][1], mode='all')
		common_neighbor[i] = len(set(neighborhood_a).intersection(
								                     set(neighborhood_b)))
	invalid_common_neighbor = np.empty(len(citation_links))
	for i in range(len(citation_links)):
		neighborhood_a = paper_graph.neighbors(citation_links[i][0], mode='in')
		neighborhood_b = paper_graph.neighbors(citation_links[i][1], mode='out')
		invalid_common_neighbor[i] = len(set(neighborhood_a).intersection(
								                     set(neighborhood_b)))
	return common_neighbor - invalid_common_neighbor

def get_common_neighbor_as_feature(paper_graph, citation_links):
	"""
	For each pair of papers (paper_a, paper_b), compute the number of valid 
	common neighbors between paper_a and paper_b.
	"""
#	from sets import Set
	common_neighbor = np.empty(len(citation_links))
	for i in range(len(citation_links)):
		neighborhood_a = paper_graph.neighbors(citation_links[i][0], mode='all')
		neighborhood_b = paper_graph.neighbors(citation_links[i][1], mode='all')
		common_neighbor[i] = len(set(neighborhood_a).intersection(
								                     set(neighborhood_b)))
	return common_neighbor

def get_adamic_adar_index_as_feature(paper_graph, citation_links):
	degrees = paper_graph.degree()
#	from sets import Set
	aa = np.empty(len(citation_links))
	for i in range(len(citation_links)):
		neighborhood_a = paper_graph.neighbors(citation_links[i][0], mode='all')
		neighborhood_b = paper_graph.neighbors(citation_links[i][1], mode='all')
		common_neighbors = set(neighborhood_a).intersection(
								                     set(neighborhood_b))
		aa[i] = np.sum([1/np.log(degrees[neighbor]) 
						for neighbor in common_neighbors])
	return aa

def get_resource_allocation_index_as_feature(paper_graph, citation_links):
	degrees = paper_graph.degree()
#	from sets import Set
	ra = np.empty(len(citation_links))
	for i in range(len(citation_links)):
		neighborhood_a = paper_graph.neighbors(citation_links[i][0], mode='all')
		neighborhood_b = paper_graph.neighbors(citation_links[i][1], mode='all')
		common_neighbors = set(neighborhood_a).intersection(
								                     set(neighborhood_b))
		ra[i] = np.sum([1/degrees[neighbor] for neighbor in common_neighbors])
	return ra


# ra = aa :take the sum of inverse log of degree of common_neighbor in each row
# the intuition in here is if the 2 papers has cited from the same papers and these papers and these papers is popular,
# Then it is highly possibility that it will be cited from each other

# common_neighbor : the number of common neighbor in each pair of paper

def get_common_neighbor_aa_ra(paper_graph, citation_links):
	degrees = paper_graph.degree()
#	from sets import Set
	ra = np.empty(len(citation_links))
	aa = np.empty(len(citation_links))
	common_neighbor = np.empty(len(citation_links))
	for i in range(len(citation_links)):
		neighborhood_a = paper_graph.neighbors(citation_links[i][0], mode='all')
		neighborhood_b = paper_graph.neighbors(citation_links[i][1], mode='all')
		common_neighbors = set(neighborhood_a).intersection(
								                     set(neighborhood_b))
		common_neighbor[i] = len(common_neighbors)
		aa[i] = np.sum([1/np.log(degrees[neighbor])       #take the sum of inverse log of degree of common_neighbor in each row
						for neighbor in common_neighbors])
		ra[i] = np.sum([1/np.log(degrees[neighbor]) 
						for neighbor in common_neighbors])
	return common_neighbor, aa, ra


def get_similarity_dice_as_feature(paper_graph, citation_links, mode='all', 
								   loops=False):
	paper_similarity_dice = np.empty(len(citation_links))
	for i in range(len(citation_links)):
		paper_similarity_dice[i] = paper_graph.similarity_dice(
			pairs=(citation_links[i][0], citation_links[i][1]), mode=mode, 
				   loops=loops)[0]
	return paper_similarity_dice

def get_similarity_jaccard_as_feature(paper_graph, citation_links, mode='all', 
									  loops=False):
	paper_similarity_jaccard = np.empty(len(citation_links))
	for i in range(len(citation_links)):
		paper_similarity_jaccard[i] = paper_graph.similarity_jaccard(pairs=(citation_links[i][0], citation_links[i][1]), mode=mode, loops=loops)[0]
	return paper_similarity_jaccard



