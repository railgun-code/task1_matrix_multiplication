#include <iostream>
#include <stdio.h>
#include <assert.h>
#include <omp.h>


void mult(double* A, double* B, double* C, int M, int N, int K, int block_size_row, int block_size_col)
{
    assert(M % block_size_row == 0 && K % block_size_col == 0);

    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++)
            C[i * N + j] = 0;
    omp_set_num_threads(120);

    #pragma omp parallel for   
    for (int ik = 0; ik < M / block_size_row; ik++)
        for (int jk = 0; jk < K / block_size_col; jk++)
            for (int i = ik * block_size_row; i < ik * block_size_row + block_size_row; i++)
                for (int k = jk * block_size_col; k < jk * block_size_col + block_size_col; k++)
                    #pragma simd
                    for (int j = 0; j < N; j++)
                        C[i * N + j] += A[i * K + k] * B[k * N + j];
}

int get_thread_count() {
    int thread_count;
#pragma omp parallel 
    {
#pragma omp single
        thread_count = omp_get_num_threads();
    }
    return thread_count;
}

void alloc_matrix(double** mat, int n, int k)
{
    (*mat) = (double*)_mm_malloc(sizeof(double) * (n * k), 64);
}

void gen_matrix(double* mat, int n, int k)
{
    srand(time(0));
    for (int i = 0; i < n; i++)
        for (int j = 0; j < k; j++)
            mat[i * k + j] = -1000.0 + rand() / 2000.0;
}

void free_matrix(double** mat)
{
    _mm_free((*mat));
}

int main(int argc, char** argv)
{
    int N, M, K;
    int block_size_row, block_size_col;

    M = 1500;
    N = 1000;
    K = 2000;
    block_size_row = 150;
    block_size_col = 200;

    double* A, * B, * C;

    alloc_matrix(&A, M, K);
    alloc_matrix(&B, K, N);
    alloc_matrix(&C, M, N);

    gen_matrix(A, M, K);
    gen_matrix(B, K, N);

    mult(A, B, C, M, N, K, block_size_row, block_size_col);

    free_matrix(&A);
    free_matrix(&B);
    free_matrix(&C);

    return 0;
}

