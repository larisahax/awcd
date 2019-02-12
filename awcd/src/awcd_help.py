import numpy as np


def clustering_from_weights_sparse(adjmatrix):
	n = adjmatrix.get_shape()[0]
	clusters = []
	points = range(n)
	neighbours = np.asarray(adjmatrix.sum(axis=1).T)[0, :]
	candidates = np.argsort(neighbours)
	for i in reversed(range(len(candidates))):
		if i not in points:
			continue
		cluster_generator = candidates[i]
		all_cliques = ()
		for i in range(len(points)):
			if adjmatrix[cluster_generator, i] == 1:
				all_cliques += tuple([i])
		clusters.append([points[i] for i in range(len(points)) if i in all_cliques])
		ind = [i for i in range(len(points)) if i not in all_cliques]
		if len(ind) == 0:
			break
		points = [points[i] for i in range(len(points)) if i not in all_cliques]
	clusters = {i: clusters[i] for i in range(len(clusters))}
	return clusters
