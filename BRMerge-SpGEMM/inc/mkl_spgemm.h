#ifndef _Z_MKL_SPGEMM_
#define _Z_MKL_SPGEMM_

#include "common.h"
#include "CSR.h"
#include "mkl.h"

void mkl(int *arpt, int *acol, mdouble *aval,
            int *brpt, int *bcol, mdouble *bval,
            int **crpt_, int **ccol_, mdouble **cval_,
            int M, int K, int N, mint *cnnz_){

    sparse_matrix_t csr_A = nullptr, csr_B = nullptr, csr_C = nullptr;
    MKL_INT status;
    // step1 mkl_create_csr
    status = mkl_sparse_d_create_csr( &csr_A, SPARSE_INDEX_BASE_ZERO, M, K, arpt, arpt+1, acol, aval );
    assert(status == SPARSE_STATUS_SUCCESS);
    status = mkl_sparse_d_create_csr( &csr_B, SPARSE_INDEX_BASE_ZERO, K, N, brpt, brpt+1, bcol, bval );
    assert(status == SPARSE_STATUS_SUCCESS);

    // step 2 mkl_spmm
    status = mkl_sparse_spmm( SPARSE_OPERATION_NON_TRANSPOSE, csr_A, csr_B, &csr_C );
    assert(status == SPARSE_STATUS_SUCCESS);

    // step 3 mkl convert
    status = mkl_sparse_convert_csr(csr_C, SPARSE_OPERATION_NON_TRANSPOSE, &csr_C);
    assert(status == SPARSE_STATUS_SUCCESS);

    // step 4 mkl order
    status = mkl_sparse_order(csr_C);
    assert(status == SPARSE_STATUS_SUCCESS);

    // step 5 mkl export
    MKL_INT *pointerB_C, *pointerE_C, *col_index_C;
    double *csr_values_C;
    sparse_index_base_t indexing;
    status = mkl_sparse_d_export_csr(csr_C, &indexing, &M, &N, &pointerB_C, &pointerE_C, &col_index_C, &csr_values_C );
    assert(status == SPARSE_STATUS_SUCCESS);
    *cnnz_ = pointerE_C[M-1];
    
    // copy data from MKL's internal memory space to the CSR space for comparison with other library
    int cnnz = *cnnz_;
    *crpt_ = new mint [M + 1];
    *ccol_ = new mint [cnnz];
    *cval_ = new mdouble [cnnz];
    mint *crpt = *crpt_;
    mint *ccol = *ccol_;
    mdouble *cval = *cval_;
    memcpy(crpt, pointerB_C, M*sizeof(mint));
    crpt[M] = cnnz;
    memcpy(ccol, col_index_C, cnnz * sizeof(mint));
    memcpy(cval, csr_values_C, cnnz * sizeof(mdouble));
    // destroy MKL's memory
    status = mkl_sparse_destroy(csr_C);
    assert(status == SPARSE_STATUS_SUCCESS);
    status = mkl_sparse_destroy(csr_B);
    assert(status == SPARSE_STATUS_SUCCESS);
    status = mkl_sparse_destroy(csr_A);
    assert(status == SPARSE_STATUS_SUCCESS);
}


inline void mkl(const CSR& A, const CSR& B, CSR& C){
    C.M = A.M;
    C.N = B.N;
    mkl(A.rpt, A.col, A.val, B.rpt, B.col, B.val, &C.rpt, &C.col, &C.val, A.M, A.N, B.N, &C.nnz);
}


#endif
