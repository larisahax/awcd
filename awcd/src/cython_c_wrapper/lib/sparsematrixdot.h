#ifndef EXAMPLES_H
#define EXAMPLES_H


void sparse_dot(int *ptr_a_view,
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
          );

void sparse_dot_mask(int *ptr_a_view,
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
          );

#endif