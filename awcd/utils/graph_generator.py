import numpy as np
from scipy.sparse import csr_matrix
    
def generate_sbm(n_list, theta):
    k = theta.shape[0]
    n = sum(n_list)
    A = np.zeros((n, n))
    A_true = np.zeros((n, n))
    start = [sum(n_list[:i]) for i in range(len(n_list))]
    for i in range(k):
        A_true[start[i]:start[i] + n_list[i], start[i]:start[i] + n_list[i]] = np.ones((n_list[i], n_list[i]))
        for j in range(i, k, 1):
            A[start[i]:start[i] + n_list[i], start[j]:start[j] + n_list[j]] = np.random.binomial(1, theta[i,j], size = (n_list[i], n_list[j]))
    A[np.tril_indices(n)] = 0
    A += A.T
    np.fill_diagonal(A, 1)
    A = csr_matrix(A)
    A_true = {}

    for c_i in range(len(n_list)):
        comm_start = sum(n_list[:c_i])
        for i in range(comm_start, comm_start + n_list[c_i]):
            A_true[i] = list(range(comm_start, comm_start + n_list[c_i]))
    return A_true, A, sum(n_list)
