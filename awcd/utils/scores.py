from sklearn import metrics
import numpy as np
import networkx as nx
from src.awcd_help import clustering_from_weights_sparse

def NMI_sum(true, answer):
	NMI = metrics.normalized_mutual_info_score(true, answer)
	return NMI


def NMI(dict_1, dict_2, n):
	# simple version, we assume the partiotions cover all nodes
	answer_1 = np.zeros(n, dtype=int)
	answer_2 = np.zeros(n, dtype=int)
	for comm_i in dict_1:
		answer_1[dict_1[comm_i]] = comm_i
	for comm_i in dict_2:
		answer_2[dict_2[comm_i]] = comm_i
	nmi = NMI_sum(answer_1,  answer_2)
	return nmi


def modularity_calc(W, sparse=False, control_95=1, n=0):
	clusters = clustering_from_weights_sparse(W)
	n_l = len(W.indices)
	edges = np.zeros((n_l, 2))
	for i in range(n):
		edges[W.indptr[i]: W.indptr[i + 1], 0] = i
	edges[:, 1] = W.indices
	edges = [tuple(edges[i, :]) for i in range(len(edges))]
	G = nx.DiGraph(edges)
	G = G.to_undirected()
	partition = np.zeros((n,), dtype='int')
	for i in range(len(clusters)):
		partition[clusters[i]] = i
	partition = dict(zip(range(n), partition))
	mod = modularity(partition, G)
	return mod


def modularity(partition, graph, weight='weight'):
	"""Compute the modularity of a partition of a graph

    Parameters
    ----------
    partition : dict
       the partition of the nodes, i.e a dictionary where keys are their nodes
       and values the communities
    graph : networkx.Graph
       the networkx graph which is decomposed
    weight : str, optional
        the key in graph to use as weight. Default to 'weight'


    Returns
    -------
    modularity : float
       The modularity

    Raises
    ------
    KeyError
       If the partition is not a partition of all graph nodes
    ValueError
        If the graph has no link
    TypeError
        If graph is not a networkx.Graph

    References
    ----------
    .. 1. Newman, M.E.J. & Girvan, M. Finding and evaluating community
    structure in networks. Physical Review E 69, 26113(2004).
    """
	if type(graph) != nx.Graph:
		raise TypeError("Bad graph type, use only non directed graph")

	inc = dict([])
	deg = dict([])
	links = graph.size(weight=weight)
	if links == 0:
		return 0
		raise ValueError("A graph without link has an undefined modularity")

	for node in graph:
		com = partition[node]
		deg[com] = deg.get(com, 0.) + graph.degree(node, weight=weight)
		for neighbor, datas in graph[node].items():
			edge_weight = datas.get(weight, 1)
			if partition[neighbor] == com:
				if neighbor == node:
					inc[com] = inc.get(com, 0.) + float(edge_weight)
				else:
					inc[com] = inc.get(com, 0.) + float(edge_weight) / 2.

	res = 0.
	for com in set(partition.values()):
		res += (inc.get(com, 0.) / links) - \
		       (deg.get(com, 0.) / (2. * links)) ** 2
	return res
