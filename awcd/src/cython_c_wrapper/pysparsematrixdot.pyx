import numpy as np
from numpy cimport ndarray
cimport numpy as np
from libc.stdlib cimport malloc, free

DTYPE = np.intc

np.import_array()


cdef extern from "sparsematrixdot.h":
    void sparse_dot(
            int *ptr_a_view,
            int *col_a_view,
            int *data_a,
            int *ptr_b,
            int *data_b,
            int *col_b,
            int n,
            int **result_i_data,
            int **result_i_col,
            int *size_c,
            int thread_number
    )
cdef extern from "sparsematrixdot.h":
    void sparse_dot_mask(
            int *ptr_a_view,
            int *col_a_view,
            int *data_a,
            int *ptr_b,
            int *data_b,
            int *col_b,
            int n,
            int **result_i_data,
            int **result_i_col,
            int *size_c,
            int thread_number,
            int *data_mask,
            int *col_mask,
            int *ptr_mask
    )


def sparse_matrix_dot(ndarray[int, ndim=1] data_a_np not None,
                      ndarray[int, ndim=1] col_a not None,
                      ndarray[int, ndim=1] ptr_a not None,
                      ndarray[int, ndim=1] data_b_np not None,
                      ndarray[int, ndim=1] col_b_np not None,
                      ndarray[int, ndim=1] ptr_b_np not None,
                      ndarray[int, ndim=1] mask_data_np,
                      ndarray[int, ndim=1] mask_col_np,
                      ndarray[int, ndim=1] mask_ptr_np,
                      int n,
                      int thread_number,
                      int mask):
    # A * B = C
    ''''''
    cdef Py_ssize_t i
    cdef Py_ssize_t j
    cdef Py_ssize_t j_index
    cdef Py_ssize_t t_1
    cdef Py_ssize_t t_11
    cdef Py_ssize_t t_12
    cdef Py_ssize_t t_13
    cdef Py_ssize_t t_2
    cdef Py_ssize_t t_20
    cdef Py_ssize_t t_21
    cdef Py_ssize_t ptr_a_i
    cdef Py_ssize_t ptr_a_i_1
    cdef Py_ssize_t u
    cdef Py_ssize_t s
    cdef Py_ssize_t k
    cdef Py_ssize_t t
    cdef Py_ssize_t e
    cdef Py_ssize_t m
    cdef Py_ssize_t x
    cdef Py_ssize_t ind_1
    cdef Py_ssize_t l
    cdef Py_ssize_t result_m
    cdef Py_ssize_t val_j
    cdef Py_ssize_t ptr_b_j
    cdef Py_ssize_t ptr_b_j_1
    cdef Py_ssize_t size_c_i
    cdef Py_ssize_t iter
    cdef C_size

    cdef int* col_a_view = &col_a[0]
    cdef int* ptr_a_view = &ptr_a[0]
    cdef int* col_b = &col_b_np[0]
    cdef int* data_b = &data_b_np[0]
    cdef int* data_a = &data_a_np[0]
    cdef int* ptr_b = &ptr_b_np[0]
    cdef int* data_mask = &mask_data_np[0]
    cdef int* col_mask = &mask_col_np[0]
    cdef int* ptr_mask = &mask_ptr_np[0]


    cdef int* b_col_view
    cdef int* b_data_view

    cdef int* result_i_data
    cdef int* result_i_col



    cdef np.npy_intp shape[1]
    cdef int* size_c_is = <int*> malloc(sizeof(int) * n)


    cdef int** result_i_datas = <int**> malloc(sizeof(int*) * n)
    cdef int** result_i_cols = <int**> malloc(sizeof(int*) * n)

    if mask == 1:
        sparse_dot_mask(
                ptr_a_view,
                col_a_view,
                data_a,
                ptr_b,
                data_b,
                col_b,
                n,
                result_i_datas,
                result_i_cols,
                size_c_is,
                thread_number,
                data_mask,
                col_mask,
                ptr_mask
            )
    else:
        sparse_dot(
            ptr_a_view,
            col_a_view,
            data_a,
            ptr_b,
            data_b,
            col_b,
            n,
            result_i_datas,
            result_i_cols,
            size_c_is,
            thread_number
        )

    C_size = 0
    for i in range(n):
        C_size += size_c_is[i]
    cdef int* C_data = <int*> malloc(sizeof(int) * C_size)
    cdef int* C_col = <int*> malloc(sizeof(int) * C_size)
    cdef int* C_ptr = <int*> malloc(sizeof(int) * (n + 1))
    C_ptr[0] = 0
    iter = 0
    for i in range(n):
        result_i_data = result_i_datas[i]
        result_i_col = result_i_cols[i]
        size_c_i = size_c_is[i]
        for j in range(size_c_i):
            C_data[iter] = result_i_data[j]
            C_col[iter] = result_i_col[j]
            iter += 1
        C_ptr[i + 1] = C_ptr[i] + size_c_i
        free(result_i_data)
        free(result_i_col)
    free(result_i_datas)
    free(result_i_cols)
    free(size_c_is)
    shape[0] = <np.npy_intp> C_size
    C_data_np = np.PyArray_SimpleNewFromData(1, shape, np.NPY_INT, <void *> C_data)
    np.PyArray_UpdateFlags(C_data_np, C_data_np.flags.num | np.NPY_OWNDATA)
    C_col_np = np.PyArray_SimpleNewFromData(1, shape, np.NPY_INT, <void *> C_col)
    np.PyArray_UpdateFlags(C_col_np, C_col_np.flags.num | np.NPY_OWNDATA)
    C_size = n + 1
    shape[0] = <np.npy_intp> C_size
    C_ptr_np = np.PyArray_SimpleNewFromData(1, shape, np.NPY_INT, <void *> C_ptr)
    np.PyArray_UpdateFlags(C_ptr_np, C_ptr_np.flags.num | np.NPY_OWNDATA)
    return C_data_np, C_col_np, C_ptr_np