#ifndef _Z_CONVERT_H_
#define _Z_CONVERT_H_


// csr to csc
template <typename IT, typename NT>
void convert(CSC<IT, NT> &csc, const CSR<IT, NT>& csr){
    csc.make_empty();
    // csr_to_coo
    IT M = csr.rows;
    IT N = csr.cols;
    IT nnz = csr.nnz;
    IT *I = new IT [nnz];
    IT *J = new IT [nnz];
    NT *values = new NT [nnz];
    IT cnt = 0;
    for(IT i = 0; i < M; i++){
        for(IT j = csr.rowptr[i]; j < csr.rowptr[i+1]; j++){
            I[cnt] = i;
            J[cnt] = csr.colids[j];
            values[cnt++] = csr.values[j];
        }
    }
    // coo_sort in csc order
    Pair<long, NT> *p = new Pair<long, NT> [nnz];
    for(IT i = 0; i < nnz; i++){
        p[i].ind = I[i] + (long)M * J[i];
        p[i].val = values[i];
    }
    std::sort(p, p + nnz);
    for(IT i = 0; i < nnz; i++){
        I[i] = p[i].ind % M;
        J[i] = p[i].ind / M;
        values[i] = p[i].val;
    }
    delete [] p;
    // coo_to_csc
    csc.colptr = my_malloc<IT>(N + 1);
    csc.rowids = my_malloc<IT>(nnz);
    csc.values = my_malloc<NT>(nnz);
    memset(csc.colptr, 0, sizeof(IT) * (N + 1));
    for(IT i = 0; i < nnz; i++){
        csc.colptr[J[i] + 1] ++;
    }
    for(IT i = 1; i <= N; i++){
        csc.colptr[i] += csc.colptr[i - 1];
    }
    memcpy(csc.rowids, I, nnz * sizeof(IT));
    memcpy(csc.values, values, nnz * sizeof(NT));
    csc.rows = M;
    csc.cols = N;
    csc.nnz = nnz;
    delete [] I;
    delete [] J;
    delete [] values;
}


// csc to csr
template <typename IT, typename NT>
void convert(CSR<IT, NT> &csr, const CSC<IT, NT>& csc){
    csr.make_empty();
    // csc_to_coo
    IT M = csc.rows;
    IT N = csc.cols;
    IT nnz = csc.nnz;
    IT *I = new IT [nnz];
    IT *J = new IT [nnz];
    NT *values = new NT [nnz];
    IT cnt = 0;
    for(IT i = 0; i < N; i++){
        for(IT j = csc.colptr[i]; j < csc.colptr[i+1]; j++){
            I[cnt] = csc.rowids[j];
            J[cnt] = i;
            values[cnt++] = csc.values[j];
        }
    }
    // coo_sort in csr order
    Pair<long, NT> *p = new Pair<long, NT> [nnz];
    for(IT i = 0; i < nnz; i++){
        p[i].ind = (long)N * I[i] + J[i];
        p[i].val = values[i];
    }
    std::sort(p, p + nnz);
    for(IT i = 0; i < nnz; i++){
        I[i] = p[i].ind / N;
        J[i] = p[i].ind % N;
        values[i] = p[i].val;
    }
    delete [] p;
    // coo_to_csr
    csr.rowptr = my_malloc<IT>(M + 1);
    csr.colids = my_malloc<IT>(nnz);
    csr.values = my_malloc<NT>(nnz);
    memset(csr.rowptr, 0, sizeof(IT) * (M + 1));
    for(IT i = 0; i < nnz; i++){
        csr.rowptr[I[i] + 1] ++;
    }
    for(IT i = 1; i <= M; i++){
        csr.rowptr[i] += csr.rowptr[i - 1];
    }
    memcpy(csr.colids, J, nnz * sizeof(IT));
    memcpy(csr.values, values, nnz * sizeof(NT));
    csr.rows = M;
    csr.cols = N;
    csr.nnz = nnz;
    delete [] I;
    delete [] J;
    delete [] values;
}



#endif
