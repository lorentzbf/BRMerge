#ifndef YUSUKE_BIN_H
#define YUSUKE_BIN_H

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <immintrin.h>
#include <algorithm>

#include "utility.h"

/*
 * Manage the group
 * prepare hash table for each group
 */
template <class IT, class NT>
class BIN
{
public:
    BIN(): total_intprod(0), max_intprod(0), max_nz(0), thread_num(omp_get_max_threads())
    {
    }
    BIN(IT rows): total_intprod(0), max_intprod(0), max_nz(0), thread_num(omp_get_max_threads()), min_ht_size(8)
    {
        assert(rows != 0);
        row_nz = my_malloc<IT>(rows);
        rows_offset = my_malloc<IT>(thread_num + 1);
        bin_id = my_malloc<char>(rows);
        local_hash_table_id = my_malloc<IT*>(thread_num);
        local_hash_table_val = my_malloc<NT*>(thread_num);
    }
    BIN(IT rows, IT ht_size): total_intprod(0), max_intprod(0), max_nz(0), thread_num(omp_get_max_threads()), min_ht_size(ht_size)
    {
        assert(rows != 0);
        row_nz = my_malloc<IT>(rows);
        rows_offset = my_malloc<IT>(thread_num + 1);
        bin_id = my_malloc<char>(rows);
        local_hash_table_id = my_malloc<IT*>(thread_num);
        local_hash_table_val = my_malloc<NT*>(thread_num);
    }
    ~BIN() {
        my_free<IT>(row_nz);
        my_free<IT>(rows_offset);
        my_free<char>(bin_id);
#pragma omp parallel
        {
            int tid = omp_get_thread_num();
            my_free<IT>(local_hash_table_id[tid]);
            my_free<NT>(local_hash_table_val[tid]);
        }
        my_free<IT*>(local_hash_table_id);
        my_free<NT*>(local_hash_table_val);
    }

    void set_max_bin(const IT *arpt, const IT *acol, const IT *brpt, const IT rows, const IT cols);
    void set_min_bin(const IT rows, const IT cols);
    void create_local_hash_table(const IT cols);
    
    void set_intprod_num(const IT *arpt, const IT *acol, const IT *brpt, const IT rows);
    void set_rows_offset(const IT rows);
    void set_bin_id(const IT rows, const IT cols, const IT min);

    long long int total_intprod;
    IT max_intprod;
    IT max_nz;
    IT thread_num;
    IT min_ht_size;

    IT *row_nz; // the number of flop or non-zero elements of output matrix
    IT *rows_offset; // offset for row_nz
    char *bin_id;
    IT **local_hash_table_id;
    NT **local_hash_table_val;
};

/* Count the number of intermediate products per row (= flop / 2) */
template <class IT, class NT>
inline void BIN<IT, NT>::set_intprod_num(const IT *arpt, const IT *acol, const IT *brpt, const IT rows)
{
#pragma omp parallel
    {
        IT each_int_prod = 0;
#pragma omp for
        for (IT i = 0; i < rows; ++i) {
            IT nz_per_row = 0;
            for (IT j = arpt[i]; j < arpt[i + 1]; ++j) {
                nz_per_row += brpt[acol[j] + 1] - brpt[acol[j]];
            }
            row_nz[i] = nz_per_row;
            each_int_prod += nz_per_row;
        }
#pragma omp atomic
        total_intprod += each_int_prod;
    }
}

/* Get total number of floating operations and average
 * then, use it for assigning rows to thread as the amount of work is equally distributed
 */
template <class IT, class NT>
inline void BIN<IT, NT>::set_rows_offset(const IT rows)
{
    IT *ps_row_nz = my_malloc<IT>(rows + 1);

    /* Prefix sum of #intermediate products */
    scan(row_nz, ps_row_nz, rows + 1);

    IT average_intprod = (total_intprod + thread_num - 1) / thread_num;
    // long long int average_intprod = total_intprod / thread_num;

    /* Search end point of each range */
    rows_offset[0] = 0;
#pragma omp parallel
    {
        int tid = omp_get_thread_num();
        long long int end_itr = (lower_bound(ps_row_nz, ps_row_nz + rows + 1, average_intprod * (tid + 1))) - ps_row_nz;
        rows_offset[tid + 1] = end_itr;
        // if (tid == thread_num - 1) rows_offset[tid + 1] = rows;
    }
    rows_offset[thread_num] = rows;
    my_free<IT>(ps_row_nz);
}

/*
 * Prepare hash table for each thread_num
 * once allocate memory space for hash table, the thread reuse it for each row
 */
template <class IT, class NT>
inline void BIN<IT, NT>::create_local_hash_table(const IT cols)
{
#pragma omp parallel
    {
        int tid = omp_get_thread_num();
        IT ht_size = 0;
        /* Get max size of hash table */
        for (IT j = rows_offset[tid]; j < rows_offset[tid + 1]; ++j) {
            if (ht_size < row_nz[j]) ht_size = row_nz[j];
        }
        /* the size of hash table is aligned as 2^n */
        if (ht_size > 0) {
            if (ht_size > cols) ht_size = cols;
            int k = min_ht_size;
            while (k < ht_size) {
                k <<= 1;
            }
            ht_size = k;
        }
        local_hash_table_id[tid] = my_malloc<IT>(ht_size);
        local_hash_table_val[tid] = my_malloc<NT>(ht_size);
    }    
}

/*
 * Precompute how many entries each row requires for the hash table
 * the size is 2^bin_id
 */
template <class IT, class NT>
inline void BIN<IT, NT>::set_bin_id(const IT rows, const IT cols, const IT min)
{
    IT i;
#pragma omp parallel for
    for (i = 0; i < rows; ++i) {
        IT j;
        IT nz_per_row = row_nz[i];
        if (nz_per_row > cols) nz_per_row = cols;
        if (nz_per_row == 0) {
            bin_id[i] = 0;
        }
        else {
            j = 0;
            while (nz_per_row > (min << j)) {
                j++;
            }
            bin_id[i] = j + 1;
        }
    }
}

/* grouping and preparing hash table based on the number of floating operations */
template <class IT, class NT>
inline void BIN<IT, NT>::set_max_bin(const IT *arpt, const IT *acol, const IT *brpt, const IT rows, const IT cols)
{
    set_intprod_num(arpt, acol, brpt, rows);
    set_rows_offset(rows);
    set_bin_id(rows, cols, min_ht_size);
}

/* Reset the size of hash table which each row requires */
template <class IT, class NT>
inline void BIN<IT, NT>::set_min_bin(const IT rows, const IT cols)
{
    set_bin_id(rows, cols, min_ht_size);
}

#endif