#ifndef _Z_MKL_SPGEMM_
#define _Z_MKL_SPGEMM_

#include "common.h"
#include "CSR.h"
#include "mkl.h"
#include "Timings.h"
long compute_flop(mint *arpt, mint *acol, mint *brpt, mint M, mint* nnzrA, mint *floprC){
    long total_flop = 0;
#pragma omp parallel
{
    int thread_flop = 0;
#pragma omp for
    for(mint i = 0; i < M; i++){
        int local_sum = 0;
        nnzrA[i] = arpt[i+1] - arpt[i];
        for(mint j = arpt[i]; j < arpt[i+1]; j++){
            local_sum += brpt[acol[j]+1] - brpt[acol[j]];
        }
        floprC[i] = local_sum;
        thread_flop += local_sum;
    }
#pragma omp critical
{
    total_flop += thread_flop;
}
}
    return total_flop;
}

long compute_flop(const CSR<int, double>& A, const CSR<int, double>& B){
    mint *nnzrA = new mint [A.rows];
    mint *row_flop = new mint [A.rows];
    long flop = compute_flop(A.rowptr, A.colids, B.rowptr, A.rows, nnzrA, row_flop);
    delete [] row_flop;
    delete [] nnzrA;
    return flop;
}

void mkl(int *arpt, int *acol, mdouble *aval,
            int *brpt, int *bcol, mdouble *bval,
            int **crpt_, int **ccol_, mdouble **cval_,
            int M, int K, int N, mint *cnnz_, Timings &timing){

    sparse_matrix_t csr_A = nullptr, csr_B = nullptr, csr_C = nullptr;
    MKL_INT status;
    double t0, t1;
    // step1 mkl_create_csr
    t0 = t1 = fast_clock_time();
    status = mkl_sparse_d_create_csr( &csr_A, SPARSE_INDEX_BASE_ZERO, M, K, arpt, arpt+1, acol, aval );
    assert(status == SPARSE_STATUS_SUCCESS);
    status = mkl_sparse_d_create_csr( &csr_B, SPARSE_INDEX_BASE_ZERO, K, N, brpt, brpt+1, bcol, bval );
    assert(status == SPARSE_STATUS_SUCCESS);
    timing.create = fast_clock_time() - t0;

    // step 2 mkl_spmm
    t0 = fast_clock_time();
    status = mkl_sparse_spmm( SPARSE_OPERATION_NON_TRANSPOSE, csr_A, csr_B, &csr_C );
    assert(status == SPARSE_STATUS_SUCCESS);
    timing.spmm = fast_clock_time() - t0;

    // step 3 mkl convert
    t0 = fast_clock_time();
    status = mkl_sparse_convert_csr(csr_C, SPARSE_OPERATION_NON_TRANSPOSE, &csr_C);
    assert(status == SPARSE_STATUS_SUCCESS);
    timing.convert = fast_clock_time() - t0;

    // step 4 mkl order
    t0 = fast_clock_time();
    status = mkl_sparse_order(csr_C);
    assert(status == SPARSE_STATUS_SUCCESS);
    timing.order = fast_clock_time() - t0;

    // step 5 mkl export
    t0 = fast_clock_time();
    MKL_INT *pointerB_C, *pointerE_C, *col_index_C;
    double *csr_values_C;
    sparse_index_base_t indexing;
    status = mkl_sparse_d_export_csr(csr_C, &indexing, &M, &N, &pointerB_C, &pointerE_C, &col_index_C, &csr_values_C );
    assert(status == SPARSE_STATUS_SUCCESS);
    *cnnz_ = pointerE_C[M-1];
    timing.export_csr = fast_clock_time() - t0;
    timing.total = fast_clock_time() - t1;
    
    // copy data from MKL's internal memory space to the CSR space for comparison with other library
    int cnnz = *cnnz_;
    //*crpt_ = new mint [M + 1];
    //*ccol_ = new mint [cnnz];
    //*cval_ = new mdouble [cnnz];
    *crpt_ = my_malloc<mint>(M + 1);
    *ccol_ = my_malloc<mint>(cnnz);
    *cval_ = my_malloc<mdouble>(cnnz);
    mint *crpt = *crpt_;
    mint *ccol = *ccol_;
    mdouble *cval = *cval_;
    memcpy(crpt, pointerB_C, M*sizeof(mint));
    crpt[M] = cnnz;
    memcpy(ccol, col_index_C, cnnz * sizeof(mint));
    memcpy(cval, csr_values_C, cnnz * sizeof(mdouble));
    // destroy MKL's memory
    t0 = fast_clock_time();
    status = mkl_sparse_destroy(csr_C);
    assert(status == SPARSE_STATUS_SUCCESS);
    status = mkl_sparse_destroy(csr_B);
    assert(status == SPARSE_STATUS_SUCCESS);
    status = mkl_sparse_destroy(csr_A);
    assert(status == SPARSE_STATUS_SUCCESS);
    timing.destroy = fast_clock_time() - t0;
    timing.total += timing.destroy;
}


inline void mkl(const CSR<int, double>& A, const CSR<int, double>& B, CSR<int, double>& C,  Timings &timing){
    C.rows = A.rows;
    C.cols = B.cols;
    mkl(A.rowptr, A.colids, A.values, B.rowptr, B.colids, B.values, &C.rowptr, &C.colids, &C.values, A.rows, A.cols, B.cols, &C.nnz, timing);
}

inline void mkl(const CSR<int, double>& A, const CSR<int, double>& B, CSR<int, double>& C){
    Timings timing;
    C.rows = A.rows;
    C.cols = B.cols;
    mkl(A.rowptr, A.colids, A.values, B.rowptr, B.colids, B.values, &C.rowptr, &C.colids, &C.values, A.rows, A.cols, B.cols, &C.nnz, timing);
}


#endif
