#ifndef Z_CSR_H_
#define Z_CSR_H_
#include <string>
#include <vector>
#include "common.h"

class CSR{
    public:
    mint M;
    mint N;
    mint nnz;
    mint *rpt;
    mint *col;
    mdouble *val;

    CSR():M(0), N(0), nnz(0), rpt(nullptr), col(nullptr), val(nullptr)
        {}
    CSR(const std::string &mtx_file);
    CSR(const CSR& A);
    CSR(const CSR& A, mint M_, mint N_, mint M_start, mint N_start);
    CSR(const CSR& A, const std::vector<int> &rows);
    ~CSR();

    void release();
    bool operator==(const CSR& A);
    CSR& operator=(const CSR& A);
    void construct(const std::string &mtx_file);   
};

#endif
