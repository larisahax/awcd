import numpy as np
import matplotlib.pylab as plt
from pylab import rcParams
import copy, time
from scipy.sparse import coo_matrix, csr_matrix, csc_matrix
import numexpr as ne
from src.cython_c_wrapper.pysparsematrixdot import sparse_matrix_dot

KL_NET = 2000

USE_NE = True
NUM_THREADS = 8

def draw_step_AWCD(windows):
    """
    Plot adjacency matrices from windows into one row of plots

    :param windows: list of 2d numpy arrays
    """
    k = len(windows)
    rcParams['figure.figsize'] = 5 * k, 5
    f, (ax) = plt.subplots(1, k)
    for i in range(k):
        if windows[i] is not None:
            ax[i].imshow(windows[i], cmap=plt.get_cmap('gray'), interpolation='nearest')
    plt.show()


def helper_shorter(A, B, C, n, mask):
    a, b, c = sparse_matrix_dot(A.data, A.indices, A.indptr, B.data, B.indices, B.indptr, C.data, C.indices, C.indptr, n, NUM_THREADS, mask)
    return csr_matrix((a, b, c), shape=(n, n))


def AWCD_step_sparce(A, l, weights, KL, params={}):
    n = A.get_shape()[0]
    print ('n', n)
    weights = weights.tocsr()
    weights.data = weights.data.astype('intc')
    A.data = A.data.astype('intc')
    if 'remove_diag' in params:
        weights.setdiag([1 * (params['remove_diag'] == False)] * n)
    if 'distance' in params:
        if params['distance'] == 'wa':
            AW_data, AW_col, AW_ptr = sparse_matrix_dot(weights.data, weights.indices, weights.indptr, A.data, A.indices, A.indptr, A.data, A.indices, A.indptr, n, NUM_THREADS, 0)
            dist_matrix = csr_matrix((AW_data, AW_col, AW_ptr), shape=(n, n))
            dist_matrix = dist_matrix + dist_matrix.transpose()
            dist_matrix.sort_indices()
        elif params['distance'] == 'ww':
            dist_matrix = helper_shorter(weights, weights, weights, n, 0)
    else:
        dist_matrix = weights.dot(weights)
    # we remove center point i from calculation and all its edges ik
    dist_matrix.data = np.minimum(dist_matrix.data, 1)

    N_aa = np.asarray(weights.sum(axis=0))[0] * 1. # numpy
    #S_ak = np.maximum(np.dot(np.dot(weights, A), weights), 1)
    weights = weights.tocsr()
    #S_ak = np.maximum(np.dot(np.dot(weights, A), weights), 1)
    #A = A.tocsr()
    if params['distance'] == 'ww':
        AW_data, AW_col, AW_ptr = sparse_matrix_dot(weights.data, weights.indices, weights.indptr, A.data, A.indices, A.indptr, A.data, A.indices, A.indptr, n, NUM_THREADS, 0)
    S_ak_data, S_ak_col, S_ak_ptr = sparse_matrix_dot(AW_data, AW_col, AW_ptr, weights.data, weights.indices, weights.indptr, dist_matrix.data, dist_matrix.indices, dist_matrix.indptr, n, NUM_THREADS, 1)
    del AW_data, AW_col, AW_ptr
    S_ak = csr_matrix((S_ak_data, S_ak_col, S_ak_ptr), shape=(n, n))
    S_ak.sort_indices()
    del A

    #S_ak.data = np.maximum(S_ak.data, 1)
    csr_indptr = S_ak.indptr
    col = S_ak.indices
    row = np.zeros_like(col, dtype=int)
    for i in range(n):
        row[csr_indptr[i]: csr_indptr[i + 1]] = i
    N_aa_row = N_aa[row]
    N_aa_col = N_aa[col]
    del N_aa

    S_a = S_ak.diagonal() # numpy
    if USE_NE:
        N_a_row = ne.evaluate("N_aa_row * N_aa_row") # numpy
    else:
        N_a_row = N_aa_row * N_aa_row

    #theta_ak = S_ak / N_ak
    S_ak_data = copy.deepcopy(S_ak.data)
    del S_ak
    #N_ak = np.maximum(np.dot(np.dot(weights, np.ones((n, n))), weights), 1)
    if USE_NE:
        q = ne.evaluate("N_aa_row * N_aa_col * 1.")
    else:
        q = N_aa_row * N_aa_col * 1.
    if USE_NE:
        N_ak = ne.evaluate("where(q > 1, q, 1)")
    else:
        N_ak = np.maximum(q, 1)
    del q

    if USE_NE:
        theta_ak_data = ne.evaluate("S_ak_data / N_ak * KL_NET")
    else:
        theta_ak_data = (S_ak_data / N_ak) * KL_NET

    theta_ak_data = (theta_ak_data).astype(int)


    theta_a = csr_matrix((theta_ak_data, col, csr_indptr), shape=(n, n)).diagonal()
    theta_a_row = theta_a[row]

    #theta_a_plus_ak = (S_ak + S_a) / (N_ak + N_a)
    S_a_row = S_a[row]
    S_a_col = S_a[col]
    del row, S_a
    N_a_plus_k = N_a_row + N_aa_col * N_aa_col
    if USE_NE:
        theta_a_plus_ak_data = ne.evaluate(
            "((S_ak_data + S_a_row) / (N_ak + N_a_row)) * KL_NET"
            )
        #theta_a_plus_k = (S_a + S_a.T) / (N_a + N_a.T)
        theta_a_plus_k = ne.evaluate("((S_a_row + S_a_col) / (N_a_plus_k)) * KL_NET").astype(int)
        theta_sum_ak = ne.evaluate("((S_ak_data + S_a_row + S_a_col) / (N_ak + N_a_plus_k)) * KL_NET").astype(int)
    else:
        theta_a_plus_ak_data = ((S_ak_data + S_a_row) / (N_ak + N_a_row)) * KL_NET
    del S_a_row, S_ak_data, S_a_col
    #N_a = np.diagonal(N_ak)[:, np.newaxis]
    theta_a_plus_ak_data = (theta_a_plus_ak_data).astype(int)
    if l >= 0:
        #T_A = N_a * KL[THETA_A, theta_a_plus_ak] + \
        #      N_ak * KL[theta_ak, theta_a_plus_ak]
        kl_1 = KL[theta_a_row, theta_a_plus_ak_data]
        kl_2 = KL[theta_ak_data, theta_a_plus_ak_data]
        del theta_a_plus_ak_data
        if USE_NE:
            T_A_data = ne.evaluate("N_a_row * kl_1 + N_aa_row * N_aa_col * kl_2")
        else:
            T_A_data = N_a_row * kl_1 + N_aa_row * N_aa_col * kl_2
        del kl_1, kl_2, N_a_row, N_aa_row, N_aa_col

        x = csc_matrix((np.arange(0, len(T_A_data), dtype=int), col, csr_indptr), shape=(n, n)).tocsr()
        csr_order_T = copy.deepcopy(x.data)
        del x

        b = T_A_data[csr_order_T]
        if USE_NE:
            T_A_T_data = T_A_data + b#ne.evaluate("where(T_A_data < b, b, T_A_data)")
        else:
            T_A_T_data = np.minimum(T_A_data, b)
        del b, T_A_data

        #I_1_a = (theta_ak <= THETA_A) * (THETA_A <= THETA_K)
        #I_1_a_data = (theta_ak_data <= theta_a_row) * (theta_a_row <= theta_a_col) * 1
        theta_a_col = theta_a[col]
        del theta_a
        if USE_NE:
            I_1_a_data = ne.evaluate(
                "where(theta_ak_data <= theta_a_row, 1, 0) * where(theta_a_row <= theta_a_col, 1, 0)"
                )
        else:
            I_1_a_data = (theta_ak_data <= theta_a_row) * (theta_a_row <= theta_a_col) * 1

        del theta_a_row, theta_a_col, theta_ak_data

        c = I_1_a_data[csr_order_T]
        del csr_order_T
        if USE_NE:
            T_A_T_data = ne.evaluate("where(T_A_T_data * (I_1_a_data + c - I_1_a_data * c) > l, 1, 0)")
        else:
            T_A_T_data = ((T_A_T_data * (I_1_a_data + c - I_1_a_data * c)) > l) * 1
        del c, I_1_a_data
        # T = T_A_T
        #weights[I] = (T[I] <= l) * 1

        T_A_T = csr_matrix((T_A_T_data, col, csr_indptr), shape=(n, n))
        del col, csr_indptr, T_A_T_data

        T_A_T = T_A_T.multiply(dist_matrix)
        T_A_T = dist_matrix - T_A_T - weights.multiply(dist_matrix)
        del dist_matrix
        weights = weights + T_A_T
        del T_A_T
        #if 'remove_diag' in params:
        weights.setdiag([1] * n)
        return weights


def AWCD_step(A, l, weights, KL, params={}):
    # first we decide which pairs of nodes we want to consider
    # if dist_matrix[i, j] > 0 then we update weight[i, j]
    if 'distance' in params:
        if params['distance'] == 'wa':
            dist_matrix = np.dot(weights, A)
        elif params['distance'] == 'ww':
            dist_matrix = np.dot(weights, weights)
    else:
        dist_matrix = np.dot(weights, weights)
    # we remove center point i from calculation and all its edges ik
    if 'remove_diag' in params:
        np.fill_diagonal(weights, 1 * (params['remove_diag'] == False))
    n = weights.shape[0]
    N_ak = np.maximum(np.dot(np.dot(weights, np.ones((n, n))), weights), 1)
    N_a = np.diagonal(N_ak)[:, np.newaxis]
    S_ak = np.dot(np.dot(weights, A), weights)
    S_a = np.diagonal(S_ak)[:, np.newaxis]
    theta_ak = S_ak / N_ak
    theta_a = np.diagonal(theta_ak)[:, np.newaxis]
    theta_a_plus_ak = (S_ak + S_a) / (N_ak + N_a)

    theta_a = (theta_a * KL_NET).astype(int)
    theta_ak = (theta_ak * KL_NET).astype(int)

    theta_a_plus_ak = (theta_a_plus_ak * KL_NET).astype(int)
    theta_a_plus_ak[S_ak == 0] = 0
    '''Test Statistics'''

    THETA_A = np.repeat(theta_a, n, axis=1)
    THETA_K = np.repeat(theta_a.T, n, axis=0)

    T_A = N_a * KL[THETA_A, theta_a_plus_ak] + \
          N_ak * KL[theta_ak, theta_a_plus_ak]
    T_A[S_ak == 0] = 0
    I = (dist_matrix + dist_matrix.T) > 0
    T_1 = np.minimum(T_A * (I > 0), T_A.T * (I > 0))
    I_1_a = (theta_ak <= THETA_A) * (THETA_A <= THETA_K)
    T = np.ones((n, n)) * -1
    T = T - T * I_1_a + T_1 * I_1_a
    T = T - T * I_1_a.T + T_1 * I_1_a.T
    I = (dist_matrix + dist_matrix.T) > 0
    weights[I] = (T[I] <= l) * 1
    np.fill_diagonal(weights, 1)
    return weights


def KL_init(params):
    m = KL_NET
    e1 = np.linspace(0, 1, m + 1)
    e = np.repeat([e1], m + 1, axis=0)
    q = e.T
    KL = e * np.log(e / q) + (1. - e) * np.log((1. - e) / (1. - q))
    KL = np.nan_to_num(KL)
    KL[KL > 100000000] = 100000000
    return KL


def init(A, params):
    rcParams['figure.figsize'] = 8, 6
    n = A.shape[0]
    weights = copy.deepcopy(A)
    KL = KL_init(params)
    K = params['steps'] if 'steps' in params else 5
    return weights, KL, K


def AWCD(A, l, K=5, sparse=False, show_step=False, show_finish=False, C_true=None, return_all_steps=False, params={}, folder_name='1'):
    """
    A - matrix of edges
    l - lambda parameter
    show_step   = True -> draw results on each step
    show_finish = True -> draw results in the end
    C_true = True -> the true matrix of weights is provided. used in evaluation and result plots.
    params = {
		'distance': 'wa', # wa (W * A) for fast calculation, 'ww' (W * W) for full calculation.
		'steps': K # number of AWCD update steps
		}
    """

    n = A.shape[0]
    if not sparse:
        try:
            A = np.array(A.todense())
        except:
            A = A
    else:
        A.data = A.data.astype('intc')
        A = A.tocsr()
    start = time.time()
    weights, KL, K = init(A, params)
    print ('init', time.time() - start)
    if show_step:
        if sparse:
            C_dense = np.zeros((n, n), dtype=int)
            print ('na', n)
            for i in C_true:
                for s in C_true[i]:
                    C_dense[s, C_true[i]] = 1
            draw_step_AWCD([C_dense, A.todense(), weights.todense()])
        else:
            exit()
            C_dense = np.zeros((n, n))
            for i in C_true:
                C_dense[i, C_true[i]] = 1
            draw_step_AWCD([C_dense, A, weights])
    # the main cycle
    if return_all_steps:
        W_all = [copy.deepcopy(weights)]
    for k in range(1, K):
        print ('k={}, l={}'.format(k, l))
        if sparse:
            start = time.time()
            weights = AWCD_step_sparce(A, l, weights, KL, params)
            w_sum = np.sum(weights.data)
            print ('sum', w_sum, w_sum * 1. / n / n)
            print ('step k in {}'.format(time.time() - start))
        else:
            weights = AWCD_step(A, l, weights, KL, params)
        if return_all_steps:
            W_all.append(copy.deepcopy(weights))
        if show_step and k < K - 1:
            if sparse:
                C_dense = np.zeros((n, n), dtype=int)
                print ('na', n)
                for i in C_true:
                    for s in C_true[i]:
                        C_dense[s, C_true[i]] = 1
                draw_step_AWCD([C_dense, A.todense(), weights.todense()])
            else:
                exit()
                C_dense = np.zeros((n, n))
                for i in C_true:
                    C_dense[i, C_true[i]] = 1
                draw_step_AWCD([C_dense, A, weights])
    if show_finish:
        draw_step_AWCD([C_true, A, weights])
    if return_all_steps:
        return W_all
    return weights
