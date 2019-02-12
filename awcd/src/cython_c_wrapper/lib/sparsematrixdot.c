#include <stdio.h>
#include <stdlib.h>

#include "sparsematrixdot.h"


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
    ) {
    int u, i, j, val_j, ptr_a_i, ptr_a_i_1, ptr_b_j, ptr_b_j_1, ptr_b_j_plus_k, x, ind_1, size_c_i, m, iter, result_m;
    int *a, *b;

    int **result;
    int *result_i;
    int *result_tid;


    result = (int**) malloc(sizeof(int*) * thread_number);
    for (i = 0; i < thread_number; i++){
        result[i] = (int *) malloc(sizeof(int) * n);
        result_i = result[i];
        for (j = 0; j < n; j++){
            result[i][j] = 0;
        }
    }


    #pragma omp parallel for default(shared) num_threads(thread_number) private(u, j, val_j, ptr_a_i, ptr_a_i_1, ptr_b_j, ptr_b_j_1, ptr_b_j_plus_k, x, ind_1, size_c_i, m, iter, result_m, a, b, result_tid)
    for (i = 0; i < n; i++){

    int tid = 0;
    //int tid = omp_get_thread_num();
    result_tid = result[tid];

    ptr_a_i = ptr_a_view[i];
    ptr_a_i_1 = ptr_a_view[i + 1];
    // calculate final row C[i, :]
    for (u = ptr_a_i; u < ptr_a_i_1; u++){
        j = col_a_view[u];
        val_j = data_a[u];
        ptr_b_j = ptr_b[j];
        ptr_b_j_1 = ptr_b[j + 1];
        for (ptr_b_j_plus_k = ptr_b_j; ptr_b_j_plus_k < ptr_b_j_1; ptr_b_j_plus_k++){
            x = data_b[ptr_b_j_plus_k] * val_j;
            ind_1 = col_b[ptr_b_j_plus_k];
            result_tid[ind_1] += x;
        }
    }

    size_c_i = 0;
    for (m = 0; m < n; m++){
        if (result_tid[m] > 0){
            size_c_i += 1;
        }
    }
    size_c[i] = size_c_i;

    result_i_data[i] = (int *) malloc(sizeof(int) * size_c_i);
    result_i_col[i] = (int *) malloc(sizeof(int) * size_c_i);
    a = result_i_data[i];
    b = result_i_col[i];


    iter = 0;
    for (m = 0; m < n; m++){
        result_m = result_tid[m];
        if (result_m > 0){
            a[iter] = result_m;
            b[iter] = m;
            iter += 1;
            result_tid[m] = 0;
        }
    }
}
    for (i = 0; i < thread_number; i++){
        free(result[i]);
    }
    free(result);
}


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
    ) {
    int u, i, j, val_j, ptr_a_i, ptr_a_i_1, ptr_b_j, ptr_b_j_1, ptr_b_j_plus_k, x, ind_1, size_c_i, iter, result_m;
    int *a, *b;

    int **result;
    int **mask;
    int *result_i;
    int *result_tid;
    int *mask_i;
    int *mask_tid;

    result = (int**) malloc(sizeof(int*) * thread_number);
    for (i = 0; i < thread_number; i++){
        result[i] = (int *) malloc(sizeof(int) * n);
        result_i = result[i];
        for (j = 0; j < n; j++){
            result_i[j] = 0;
        }
    }
    mask = (int**) malloc(sizeof(int*) * thread_number);
        for (i = 0; i < thread_number; i++){
            mask[i] = (int *) malloc(sizeof(int) * n);
            mask_i = mask[i];
            for (j = 0; j < n; j++){
                mask_i[j] = 0;
            }
        }

    #pragma omp parallel for default(shared) num_threads(thread_number) private(u, j, val_j, ptr_a_i, ptr_a_i_1, ptr_b_j, ptr_b_j_1, ptr_b_j_plus_k, x, ind_1, size_c_i, iter, result_m, a, b, mask_tid, result_tid)
    for (i = 0; i < n; i++){

    int tid = 0;
    //int tid = omp_get_thread_num();

    ptr_a_i = ptr_a_view[i];
    ptr_a_i_1 = ptr_a_view[i + 1];
    // calculate final row C[i, :]

    mask_tid = mask[tid];
    result_tid = result[tid];
    for (j = ptr_mask[i]; j < ptr_mask[i + 1]; j++){
        mask_tid[col_mask[j]] = 1;
    }
    for (u = ptr_a_i; u < ptr_a_i_1; u++){
        j = col_a_view[u];
        val_j = data_a[u];
        ptr_b_j = ptr_b[j];
        ptr_b_j_1 = ptr_b[j + 1];
        for (ptr_b_j_plus_k = ptr_b_j; ptr_b_j_plus_k < ptr_b_j_1; ptr_b_j_plus_k++){
            ind_1 = col_b[ptr_b_j_plus_k];
            if (mask_tid[ind_1] == 1){
                x = data_b[ptr_b_j_plus_k] * val_j;
                result_tid[ind_1] += x;
            }
        }
    }

    for (j = ptr_mask[i]; j < ptr_mask[i + 1]; j++){
        mask_tid[col_mask[j]] = 0;
    }
    size_c_i = 0;
    for (j = ptr_mask[i]; j < ptr_mask[i + 1]; j++){
        if (result_tid[col_mask[j]] > 0){
            size_c_i += 1;
        }
    }
    size_c[i] = size_c_i;

    result_i_data[i] = (int *) malloc(sizeof(int) * size_c_i);
    result_i_col[i] = (int *) malloc(sizeof(int) * size_c_i);
    a = result_i_data[i];
    b = result_i_col[i];


    iter = 0;
    for (j = ptr_mask[i]; j < ptr_mask[i + 1]; j++){
        result_m = result_tid[col_mask[j]];
        if (result_m > 0){
            a[iter] = result_m;
            b[iter] = col_mask[j];
            iter += 1;
            result_tid[col_mask[j]] = 0;
        }
    }
}
    for (i = 0; i < thread_number; i++){
        free(result[i]);
        free(mask[i]);
    }
    free(result);
    free(mask);
}