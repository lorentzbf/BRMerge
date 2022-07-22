#ifndef YUSUKE_HASH_SPGEMM_H
#define YUSUKE_HASH_SPGEMM_H

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <immintrin.h>
#include <algorithm>

#ifdef KNL_EXE
#include <zmmintrin.h>
#else
#include <x86intrin.h>
#endif

#include "utility.h"
#include "CSR.h"
#include "BIN.h"

#define VECTORIZE

/* SpGEMM Specific Parameters */
#define HASH_SCAL 107 // Set disjoint number to hash table size (=2^n)

#ifdef KNL_EXE
#define MIN_HT_S 16 // minimum hash table size per row in symbolic phase
#define MIN_HT_N 16 // minimum hash table size per row in numeric phase
#define VEC_LENGTH 16
#define VEC_LENGTH_BIT 4
#define VEC_LENGTH_LONG 8
#define VEC_LENGTH_LONG_BIT 3

#else
#define MIN_HT_S 8 // minimum hash table size per row in symbolic phase
#define MIN_HT_N 8 // minimum hash table size per row in numeric phase
#define VEC_LENGTH 8
#define VEC_LENGTH_BIT 3
#define VEC_LENGTH_LONG 4
#define VEC_LENGTH_LONG_BIT 2
#endif

/*
 * Symbolic phase for Hash SpGEMM.
 */
template <typename IT, typename NT>
long long int get_flop(const CSR<IT,NT> & A, const CSR<IT,NT> & B)
{
    long long int flops = 0; // total flops (multiplication) needed to generate C
    long long int tflops=0; //thread private flops

    for (IT i=0; i < A.rows; ++i) {       // for all rows of A
        long long int locmax = 0;
        for (IT j=A.rowptr[i]; j < A.rowptr[i + 1]; ++j) { // For all the nonzeros of the ith column
            long long int inner = A.colids[j]; // get the row id of B (or column id of A)
            long long int npins = B.rowptr[inner + 1] - B.rowptr[inner]; // get the number of nonzeros in A's corresponding column
            locmax += npins;
        }
        tflops += locmax;
    }
    flops += tflops;
    return (flops);
}

template <class IT, class NT>
inline void hash_symbolic_kernel(const IT *arpt, const IT *acol, const IT *brpt, const IT *bcol, BIN<IT, NT> &bin)
{
#pragma omp parallel
    {
        IT tid = omp_get_thread_num();
        IT start_row = bin.rows_offset[tid];
        IT end_row = bin.rows_offset[tid + 1];
        
        IT *check = bin.local_hash_table_id[tid];
        
        for (IT i = start_row; i < end_row; ++i) {
            IT nz = 0;
            IT bid = bin.bin_id[i];
            
            if (bid > 0) {
                IT ht_size = MIN_HT_S << (bid - 1); // determine hash table size for i-th row
                for (IT j = 0; j < ht_size; ++j) { // initialize hash table
                    check[j] = -1;
                }

                for (IT j = arpt[i]; j < arpt[i + 1]; ++j) {
                    IT t_acol = acol[j];
                    for (IT k = brpt[t_acol]; k < brpt[t_acol + 1]; ++k) {
                        IT key = bcol[k];
                        IT hash = (key * HASH_SCAL) & (ht_size - 1);
                        while (1) { // Loop for hash probing
                            if (check[hash] == key) { // if the key is already inserted, it's ok
                                break;
                            }
                            else if (check[hash] == -1) { // if the key has not been inserted yet, then it's added.
                                check[hash] = key;
                                nz++;
                                break;
                            }
                            else { // linear probing: check next entry
                                hash = (hash + 1) & (ht_size - 1); //hash = (hash + 1) % ht_size
                            }
                        }
                    }
                }
            }
            bin.row_nz[i] = nz;
        }
    }
}

#ifdef KNL_EXE
/*
 * Symbolic phase for Hash Vector SpGEMM
 * This function is optimized for 32-bit integer with AVX-512.
 */
template <class NT>
inline void hash_symbolic_vec_kernel(const int *arpt, const int *acol, const int *brpt, const int *bcol, BIN<int, NT> &bin)
{
#ifdef VECTORIZE
    const __m512i init_m = _mm512_set1_epi32(-1);
#endif        

#pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int start_row = bin.rows_offset[tid];
        int end_row = bin.rows_offset[tid + 1];
        
        int *check = bin.local_hash_table_id[tid];
        
        for (int i = start_row; i < end_row; ++i) {
#ifdef VECTORIZE
            __m512i key_m, check_m;
            __mmask16 mask_m;
#endif        
            int nz = 0;
            int bid = bin.bin_id[i];
            
            if (bid > 0) {
                int table_size = MIN_HT_S << (bid - 1); // the number of entries per table
                int ht_size = table_size >> VEC_LENGTH_BIT; // the number of chunks (1 chunk = VEC_LENGTH elments)
                for (int j = 0; j < table_size; ++j) {
                    check[j] = -1; // initialize hash table
                }
                for (int j = arpt[i]; j < arpt[i + 1]; ++j) {
                    int t_acol = acol[j];
                    for (int k = brpt[t_acol]; k < brpt[t_acol + 1]; ++k) {
                        int key = bcol[k];
                        int hash = ((key * HASH_SCAL) & (ht_size - 1)) << VEC_LENGTH_BIT;
#ifdef VECTORIZE
                        key_m = _mm512_set1_epi32(key);
#endif
                        while (1) { // Loop for hash probing
                            // check whether the key is in hash table.
#ifdef VECTORIZE
                            check_m = _mm512_load_epi32(check + hash);
                            mask_m = _mm512_cmp_epi32_mask(key_m, check_m, _MM_CMPINT_EQ);
                            if (mask_m != 0) {
                                break;
                            }
#else
                            bool flag = false;
#pragma vector
                            for (int l = 0; l < VEC_LENGTH; ++l) {
                                if (check[(hash << VEC_LENGTH_BIT) + l] == key) {
                                    flag = true;
                                }
                            }
                            if (flag) {
                                break;
                            }
#endif
                            else {
                                // If the entry with same key cannot be found, check whether the chunk is filled or not
                                int cur_nz;
#ifdef VECTORIZE
                                mask_m = _mm512_cmp_epi32_mask(check_m, init_m, _MM_CMPINT_NE);
                                cur_nz = _popcnt32(mask_m);
#else
                                cur_nz = VEC_LENGTH;
#pragma vector
                                for (int l = VEC_LENGTH - 1; l >= 0; --l) {
                                    if (check[(hash << VEC_LENGTH_BIT) + l] == -1) {
                                        cur_nz = l;
                                    }
                                }
#endif
                                if (cur_nz < VEC_LENGTH) { //if it is not filled, push the entry to the table
                                    check[hash + cur_nz] = key;
                                    nz++;
                                    break;
                                }
                                else { // if is filled, check next chunk (linear probing)
                                    hash = (hash + VEC_LENGTH) & (table_size - 1);
                                }
                            }
                        }
                    }
                }
            }
            bin.row_nz[i] = nz;
        }
    }
}

/*
 * Symbolic phase for Hash Vector SpGEMM
 * This function is optimized for 64-bit integer with AVX-512.
 */
template <class NT>
inline void hash_symbolic_vec_kernel(const long long int *arpt, const long long int *acol, const long long int *brpt, const long long int *bcol, BIN<long long int, NT> &bin)
{
#ifdef VECTORIZE
    const __m512i init_m = _mm512_set1_epi64(-1);
#endif        

#pragma omp parallel
    {
        long long int tid = omp_get_thread_num();
        long long int start_row = bin.rows_offset[tid];
        long long int end_row = bin.rows_offset[tid + 1];
        
        long long int *check = bin.local_hash_table_id[tid];
        
        for (long long int i = start_row; i < end_row; ++i) {
#ifdef VECTORIZE
            __m512i key_m, check_m;
            __mmask8 mask_m;
#endif        
            long long int nz = 0;
            long long int bid = bin.bin_id[i];
            
            if (bid > 0) {
                long long int table_size = MIN_HT_S << (bid - 1); // the number of entries per table
                long long int ht_size = table_size >> VEC_LENGTH_LONG_BIT; // the number of chunks (1 chunk = VEC_LENGTH elments)
                for (long long int j = 0; j < table_size; ++j) {
                    check[j] = -1; // initialize hash table
                }
                for (long long int j = arpt[i]; j < arpt[i + 1]; ++j) {
                    long long int t_acol = acol[j];
                    for (long long int k = brpt[t_acol]; k < brpt[t_acol + 1]; ++k) {
                        long long int key = bcol[k];
                        long long int hash = ((key * HASH_SCAL) & (ht_size - 1)) << VEC_LENGTH_LONG_BIT;
#ifdef VECTORIZE
                        key_m = _mm512_set1_epi64(key);
#endif
                        while (1) { // loop for hash probing
                            // check whether the key is in hash table.
#ifdef VECTORIZE
                            check_m = _mm512_load_epi64(check + hash);
                            mask_m = _mm512_cmp_epi64_mask(key_m, check_m, _MM_CMPINT_EQ);
                            if (mask_m != 0) {
                                break;
                            }
#else
                            bool flag = false;
#pragma vector
                            for (int l = 0; l < VEC_LENGTH_LONG; ++l) {
                                if (check[(hash << VEC_LENGTH_LONG_BIT) + l] == key) {
                                    flag = true;
                                }
                            }
                            if (flag) {
                                break;
                            }
#endif
                            else {
                                // If the entry with same key cannot be found, check whether the chunk is filled or not
                                long long int cur_nz;
#ifdef VECTORIZE
                                mask_m = _mm512_cmp_epi64_mask(check_m, init_m, _MM_CMPINT_NE);
                                cur_nz = _popcnt32(mask_m);
#else
                                cur_nz = VEC_LENGTH;
#pragma vector
                                for (int l = VEC_LENGTH_LONG - 1; l >= 0; --l) {
                                    if (check[(hash << VEC_LENGTH_LONG_BIT) + l] == -1) {
                                        cur_nz = l;
                                    }
                                }
#endif
                                if (cur_nz < VEC_LENGTH_LONG) { //if it is not filled, push the entry to the table
                                    check[hash + cur_nz] = key;
                                    nz++;
                                    break;
                                }
                                else { // if is filled, check next chunk (linear probing)
                                    hash = (hash + VEC_LENGTH_LONG) & (table_size - 1);
                                }
                            }
                        }
                    }
                }
            }
            bin.row_nz[i] = nz;
        }
    }
}

#else
/*
 * Symbolic phase for Hash Vector SpGEMM
 * This function is optimized for 32-bit integer with AVX2.
 */
template <class NT>
inline void hash_symbolic_vec_kernel(const int *arpt, const int *acol, const int *brpt, const int *bcol, BIN<int, NT> &bin)
{
#ifdef VECTORIZE
    const __m256i init_m = _mm256_set1_epi32(-1);
    const __m256i true_m = _mm256_set1_epi32(0xffffffff);
#endif
    
#pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int start_row = bin.rows_offset[tid];
        int end_row = bin.rows_offset[tid + 1];
        
        int *check = bin.local_hash_table_id[tid];
        
        for (int i = start_row; i < end_row; ++i) {
#ifdef VECTORIZE
            __m256i key_m, check_m;
            __m256i mask_m;
            int mask;
#endif        
            int nz = 0;
            int bid = bin.bin_id[i];
            
            if (bid > 0) {
                int table_size = MIN_HT_S << (bid - 1); // the number of entries per table
                int ht_size = table_size >> VEC_LENGTH_BIT; // the number of chunks (1 chunk = VEC_LENGTH elments)
                for (int j = 0; j < table_size; ++j) {
                    check[j] = -1; // initialize hash table
                }

                for (int j = arpt[i]; j < arpt[i + 1]; ++j) {
                    int t_acol = acol[j];
                    for (int k = brpt[t_acol]; k < brpt[t_acol + 1]; ++k) {
                        int key = bcol[k];
                        int hash = (key * HASH_SCAL) & (ht_size - 1);
#ifdef VECTORIZE
                        key_m = _mm256_set1_epi32(key);
#endif
                        while (1) { // Loop for hash probing
                            // check whether the key is in hash table.
#ifdef VECTORIZE
                            check_m = _mm256_maskload_epi32(check + (hash << VEC_LENGTH_BIT), true_m);
                            mask_m = _mm256_cmpeq_epi32(key_m, check_m);
                            mask = _mm256_movemask_epi8(mask_m);
                            if (mask != 0) {
                                break;
                            }
#else
                            bool flag = false;
#pragma simd
                            for (int l = 0; l < VEC_LENGTH; ++l) {
                                if (check[(hash << VEC_LENGTH_BIT) + l] == key) {
                                    flag = true;
                                }
                            }
                            if (flag) {
                                break;
                            }
#endif
                            else {
                                // If the entry with same key cannot be found, check whether the chunk is filled or not
                                int cur_nz;
#ifdef VECTORIZE
                                mask_m = _mm256_cmpeq_epi32(check_m, init_m);
                                mask = _mm256_movemask_epi8(mask_m);
                                cur_nz = (32 - _popcnt32(mask)) >> 2;
#else
                                cur_nz = VEC_LENGTH;
#pragma simd
                                for (int l = VEC_LENGTH - 1; l >= 0; --l) {
                                    if (check[(hash << VEC_LENGTH_BIT) + l] == -1) {
                                        cur_nz = l;
                                    }
                                }
#endif
                                if (cur_nz < VEC_LENGTH) { //if it is not filled, push the entry to the table
                                    check[(hash << VEC_LENGTH_BIT) + cur_nz] = key;
                                    nz++;
                                    break;
                                }
                                else { // if is filled, check next chunk (linear probing)
                                    hash = (hash + 1) & (ht_size - 1);
                                }
                            }
                        }
                    }
                }
            }
            bin.row_nz[i] = nz;
        }
    }
}

template <class NT>
inline void hash_symbolic_vec_kernel(const long long int *arpt, const long long int *acol, const long long int *brpt, const long long int *bcol, BIN<long long int, NT> &bin)
{
#ifdef VECTORIZE
    const __m256i init_m = _mm256_set1_epi64x(-1);
    const __m256i true_m = _mm256_set1_epi64x(0xffffffffffffffff);
#endif
    
#pragma omp parallel
    {
        long long int tid = omp_get_thread_num();
        long long int start_row = bin.rows_offset[tid];
        long long int end_row = bin.rows_offset[tid + 1];
        
        long long int *check = bin.local_hash_table_id[tid];
        
        for (long long int i = start_row; i < end_row; ++i) {
#ifdef VECTORIZE
            __m256i key_m, check_m;
            __m256i mask_m;
            int mask;
#endif        
            long long int nz = 0;
            long long int bid = bin.bin_id[i];
            
            if (bid > 0) {
                long long int table_size = MIN_HT_S << (bid - 1);
                long long int ht_size = table_size >> VEC_LENGTH_LONG_BIT;
                for (long long int j = 0; j < table_size; ++j) {
                    check[j] = -1;
                }
                
                for (long long int j = arpt[i]; j < arpt[i + 1]; ++j) {
                    long long int t_acol = acol[j];
                    for (long long int k = brpt[t_acol]; k < brpt[t_acol + 1]; ++k) {
                        long long int key = bcol[k];
                        long long int hash = (key * HASH_SCAL) & (ht_size - 1);
#ifdef VECTORIZE
                        key_m = _mm256_set1_epi64x(key);
#endif
                        while (1) {
#ifdef VECTORIZE
                            check_m = _mm256_maskload_epi64(check + (hash << VEC_LENGTH_LONG_BIT), true_m);
                            mask_m = _mm256_cmpeq_epi64(key_m, check_m);
                            mask = _mm256_movemask_epi8(mask_m);
                            if (mask != 0) {
                                break;
                            }
#else
                            bool flag = false;
#pragma simd
                            for (int l = 0; l < VEC_LENGTH_LONG; ++l) {
                                if (check[(hash << VEC_LENGTH_LONG_BIT) + l] == key) {
                                    flag = true;
                                }
                            }
                            if (flag) {
                                break;
                            }
#endif
                            else {
                                long long int cur_nz;
#ifdef VECTORIZE
                                mask_m = _mm256_cmpeq_epi64(check_m, init_m);
                                mask = _mm256_movemask_epi8(mask_m);
                                cur_nz = (32 - _popcnt32(mask)) >> 3;
#else
                                cur_nz = VEC_LENGTH_LONG;
#pragma simd
                                for (int l = VEC_LENGTH_LONG - 1; l >= 0; --l) {
                                    if (check[(hash << VEC_LENGTH_LONG_BIT) + l] == -1) {
                                        cur_nz = l;
                                    }
                                }
#endif
                                if (cur_nz < VEC_LENGTH_LONG) {
                                    check[(hash << VEC_LENGTH_LONG_BIT) + cur_nz] = key;
                                    nz++;
                                    break;
                                }
                                else {
                                    hash = (hash + 1) & (ht_size - 1);
                                }
                            }
                        }
                    }
                }
            }
            bin.row_nz[i] = nz;
        }
    }
}
#endif

// Reference function for Symbolic phase of Hash SpGEMM
template <bool vectorProbing, class IT, class NT>
inline void hash_symbolic(const IT *arpt, const IT *acol, const IT *brpt, const IT *bcol, IT *crpt, BIN<IT, NT> &bin, const IT nrow, IT *nnz)
{
    if (vectorProbing) {
        hash_symbolic_vec_kernel(arpt, acol, brpt, bcol, bin);
    }
    else {
        hash_symbolic_kernel(arpt, acol, brpt, bcol, bin);
    }
    
    /* Set row pointer of matrix C */
    scan(bin.row_nz, crpt, nrow + 1);
    *nnz = crpt[nrow];
}

/*
 * Used for sort function.
 * Elements are sorted in ascending order.
 */
template <typename IT, typename NT>
bool sort_less(const pair<IT, NT> &left,const pair<IT, NT> &right)
{
    return left.first < right.first;
}

/*
 * After calculating on each hash table, sort them in ascending order if necessary, and then store them as output matrix
 * This function is used in hash_numeric* function.
 * the actual indices of colids and values of output matrix are rpt[rowid];
 */
template <bool sortOutput, typename IT, typename NT>
inline void sort_and_store_table2mat(IT *ht_check, NT *ht_value, IT *colids, NT * values, IT nz, IT ht_size)
{
    IT index = 0;
    // Sort elements in ascending order if necessary, and store them as output matrix
    if (sortOutput) {
        vector<pair<IT, NT>> p_vec(nz);
        for (IT j = 0; j < ht_size; ++j) { // accumulate non-zero entry from hash table
            if (ht_check[j] != -1) {
                p_vec[index++] = make_pair(ht_check[j], ht_value[j]);
            }
        }
        sort(p_vec.begin(), p_vec.end(), sort_less<IT, NT>); // sort only non-zero elements
        for (IT j = 0; j < index; ++j) { // store the results
            colids[j] = p_vec[j].first;
            values[j] = p_vec[j].second;
        }
    }
    else {
        for (IT j = 0; j < ht_size; ++j) {
            if (ht_check[j] != -1) {
                colids[index] = ht_check[j];
                values[index] = ht_value[j];
                index++;
            }
        }
    }

}

/*
 * Numeric phase in Hash SpGEMM.
 */
template <bool sortOutput, typename IT, typename NT, typename MultiplyOperation, typename AddOperation>
inline void hash_numeric(const IT *arpt, const IT *acol, const NT *aval, const IT *brpt, const IT *bcol, const NT *bval, const IT *crpt, IT *ccol, NT *cval, const BIN<IT, NT> &bin, const MultiplyOperation multop, const AddOperation addop)
{
#pragma omp parallel
    {
        IT tid = omp_get_thread_num();
        IT start_row = bin.rows_offset[tid];
        IT end_row = bin.rows_offset[tid + 1];

        IT *ht_check = bin.local_hash_table_id[tid];
        NT *ht_value = bin.local_hash_table_val[tid];

        for (IT i = start_row; i < end_row; ++i) {
            
            IT bid = bin.bin_id[i];
            if (bid > 0) {
                IT offset = crpt[i];
                IT ht_size = MIN_HT_N << (bid - 1);
                for (IT j = 0; j < ht_size; ++j) {
                    ht_check[j] = -1;
                }
                for (IT j = arpt[i]; j < arpt[i + 1]; ++j) {
                    IT t_acol = acol[j];
                    NT t_aval = aval[j];
                    for (IT k = brpt[t_acol]; k < brpt[t_acol + 1]; ++k) {
                        NT t_val = multop(t_aval, bval[k]);
                        IT key = bcol[k];
                        IT hash = (key * HASH_SCAL) & (ht_size - 1);
                        while (1) { // Loop for hash probing
                            if (ht_check[hash] == key) { // key is already inserted
                                ht_value[hash] = addop(t_val, ht_value[hash]);
                                break;
                            }
                            else if (ht_check[hash] == -1) { // insert new entry
                                ht_check[hash] = key;
                                ht_value[hash] = t_val;
                                break;
                            }
                            else {
                                hash = (hash + 1) & (ht_size - 1); // (hash + 1) % ht_size
                            }
                        }
                    }
                }
                sort_and_store_table2mat<sortOutput, IT, NT>(ht_check, ht_value,
                                                             ccol + offset, cval + offset,
                                                             crpt[i + 1] - offset, ht_size);
            }
        }
    }
}

#ifdef KNL_EXE
/*
 * Numeric phase for Hash Vector SpGEMM
 * This function is optimized for 32-bit integer with AVX-512.
 */
template <bool sortOutput, typename NT, typename MultiplyOperation, typename AddOperation>
inline void hash_numeric_vec(const int *arpt, const int *acol, const NT *aval, const int *brpt, const int *bcol, const NT *bval, const int *crpt, int *ccol, NT *cval, const BIN<int, NT> &bin, MultiplyOperation multop, AddOperation addop)
{
#ifdef VECTORIZE
    const __m512i init_m = _mm512_set1_epi32(-1);
#endif        

#pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int start_row = bin.rows_offset[tid];
        int end_row = bin.rows_offset[tid + 1];

        int *ht_check = bin.local_hash_table_id[tid];
        NT *ht_value = bin.local_hash_table_val[tid];

        for (int i = start_row; i < end_row; ++i) {
#ifdef VECTORIZE
            __m512i key_m, check_m;
            __mmask16 mask_m, k_m;
#endif        
    
            int bid = bin.bin_id[i];

            if (bid > 0) {
                int offset = crpt[i];
                int table_size = MIN_HT_N << (bid - 1); // the number of entries per table
                int ht_size = table_size >> VEC_LENGTH_BIT; // the number of chunks (1 chunk = VEC_LENGTH elments)

                for (int j = 0; j < table_size; ++j) {
                    ht_check[j] = -1; // initialize hash table
                }
  
                for (int j = arpt[i]; j < arpt[i + 1]; ++j) {
                    int t_acol = acol[j];
                    NT t_aval = aval[j];
                    for (int k = brpt[t_acol]; k < brpt[t_acol + 1]; ++k) {
                        NT t_val = multop(t_aval, bval[k]);
                        int key = bcol[k];
                        int hash = ((key * HASH_SCAL) & (ht_size - 1)) << VEC_LENGTH_BIT;
#ifdef VECTORIZE
                        key_m = _mm512_set1_epi32(key);
#endif
                        while (1) { // loop for hash probing
                            // check whether the key is in hash table.
#ifdef VECTORIZE
                            check_m = _mm512_load_epi32(ht_check + hash);
                            mask_m = _mm512_cmp_epi32_mask(key_m, check_m, _MM_CMPINT_EQ);
                            if (mask_m != 0) {
                                int target = __builtin_ctz(mask_m);
                                ht_value[hash + target] += t_val;
                                break;
                            }
#else
                            int flag = -1;
#pragma vector
                            for (int l = 0; l < VEC_LENGTH; ++l) {
                                if (ht_check[hash + l] == key) {
                                    flag = l;
                                }
                            }
                            if (flag >= 0) {
                                ht_value[hash + flag] += t_val;
                                break;
                            }
#endif
                            else {
                                // If the entry with same key cannot be found, check whether the chunk is filled or not
                                int cur_nz;
#ifdef VECTORIZE
                                mask_m = _mm512_cmp_epi32_mask(check_m, init_m, _MM_CMPINT_NE);
                                cur_nz = _popcnt32(mask_m);
#else
                                cur_nz = VEC_LENGTH;
#pragma vector
                                for (int l = 0; l < VEC_LENGTH; ++l) {
                                    if (ht_check[hash + l] == -1) {
                                        cur_nz = l;
                                        break;
                                    }
                                }
#endif
                                if (cur_nz < VEC_LENGTH) { //if it is not filled, push the entry to the table
                                    ht_check[hash + cur_nz] = key;
                                    ht_value[hash + cur_nz] = t_val;
                                    break;
                                }
                                else { // if is filled, check next chunk (linear probing)
                                    hash = (hash + VEC_LENGTH) & (table_size - 1);
                                }
                            }
                        }
                    }
                }
                sort_and_store_table2mat<sortOutput, int, NT>(ht_check, ht_value,
                                                              ccol + offset, cval + offset,
                                                              crpt[i + 1] - offset, table_size);
            }
        }
    }
}

/*
 * Numeric phase for Hash Vector SpGEMM
 * This function is optimized for 64-bit integer with AVX-512.
 */
template <bool sortOutput, typename NT, typename MultiplyOperation, typename AddOperation>
inline void hash_numeric_vec(const long long int *arpt, const long long int *acol, const NT *aval, const long long int *brpt, const long long int *bcol, const NT *bval, const long long int *crpt, long long int *ccol, NT *cval, const BIN<long long int, NT> &bin, MultiplyOperation multop, AddOperation addop)
{
#ifdef VECTORIZE
    const __m512i init_m = _mm512_set1_epi64(-1);
#endif        

#pragma omp parallel
    {
        long long int tid = omp_get_thread_num();
        long long int start_row = bin.rows_offset[tid];
        long long int end_row = bin.rows_offset[tid + 1];

        long long int *ht_check = bin.local_hash_table_id[tid];
        NT *ht_value = bin.local_hash_table_val[tid];

        for (long long int i = start_row; i < end_row; ++i) {
#ifdef VECTORIZE
            __m512i key_m, check_m;
            __mmask8 mask_m, k_m;
#endif        
    
            long long int bid = bin.bin_id[i];

            if (bid > 0) {
                long long int offset = crpt[i];
                long long int table_size = MIN_HT_N << (bid - 1);
                long long int ht_size = table_size >> VEC_LENGTH_LONG_BIT;

                for (long long int j = 0; j < table_size; ++j) {
                    ht_check[j] = -1;
                }
  
                for (long long int j = arpt[i]; j < arpt[i + 1]; ++j) {
                    long long int t_acol = acol[j];
                    NT t_aval = aval[j];
                    for (long long int k = brpt[t_acol]; k < brpt[t_acol + 1]; ++k) {
                        NT t_val = multop(t_aval, bval[k]);
                        long long int key = bcol[k];
                        long long int hash = ((key * HASH_SCAL) & (ht_size - 1)) << VEC_LENGTH_LONG_BIT;
#ifdef VECTORIZE
                        key_m = _mm512_set1_epi64(key);
#endif
                        while (1) { // loop for hash probing
#ifdef VECTORIZE
                            check_m = _mm512_load_epi64(ht_check + hash);
                            mask_m = _mm512_cmp_epi64_mask(key_m, check_m, _MM_CMPINT_EQ);
                            if (mask_m != 0) {
                                long long int target = __builtin_ctz(mask_m);
                                ht_value[hash + target] += t_val;
                                break;
                            }
#else
                            long long int flag = -1;
#pragma vector
                            for (int l = 0; l < VEC_LENGTH_LONG; ++l) {
                                if (ht_check[hash + l] == key) {
                                    flag = l;
                                }
                            }
                            if (flag >= 0) {
                                ht_value[hash + flag] += t_val;
                                break;
                            }
#endif
                            else {
                                long long int cur_nz;
#ifdef VECTORIZE
                                mask_m = _mm512_cmp_epi64_mask(check_m, init_m, _MM_CMPINT_NE);
                                cur_nz = _popcnt32(mask_m);
#else
                                cur_nz = VEC_LENGTH_LONG;
#pragma vector
                                for (IT l = 0; l < VEC_LENGTH_LONG; ++l) {
                                    if (ht_check[hash + l] == -1) {
                                        cur_nz = l;
                                        break;
                                    }
                                }
#endif
                                if (cur_nz < VEC_LENGTH_LONG) {
                                    ht_check[hash + cur_nz] = key;
                                    ht_value[hash + cur_nz] = t_val;
                                    break;
                                }
                                else {
                                    hash = (hash + VEC_LENGTH_LONG) & (table_size - 1);
                                }
                            }
                        }
                    }
                }
                sort_and_store_table2mat<sortOutput, long long int, NT>(ht_check, ht_value,
                                                                        ccol + offset, cval + offset,
                                                                        crpt[i + 1] - offset, table_size);
            }
        }
    }
}

#else
template <bool sortOutput, typename NT, typename MultiplyOperation, typename AddOperation>
inline void hash_numeric_vec(const int *arpt, const int *acol, const NT *aval, const int *brpt, const int *bcol, const NT *bval, const int *crpt, int *ccol, NT *cval, const BIN<int, NT> &bin, MultiplyOperation multop, AddOperation addop)
{
#ifdef VECTORIZE
    const __m256i init_m = _mm256_set1_epi32(-1);
    const __m256i true_m = _mm256_set1_epi32(0xffffffff);
#endif        

#pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int start_row = bin.rows_offset[tid];
        int end_row = bin.rows_offset[tid + 1];

        int *ht_check = bin.local_hash_table_id[tid];
        NT *ht_value = bin.local_hash_table_val[tid];

        for (int i = start_row; i < end_row; ++i) {
#ifdef VECTORIZE
            __m256i key_m, check_m, mask_m;
            int mask;
#endif            
            int bid = bin.bin_id[i];

            if (bid > 0) {
                int offset = crpt[i];
                int table_size = MIN_HT_N << (bid - 1);
                int ht_size = table_size >> VEC_LENGTH_BIT;

                for (int j = 0; j < table_size; ++j) {
                    ht_check[j] = -1;
                }
  
                for (int j = arpt[i]; j < arpt[i + 1]; ++j) {
                    int t_acol = acol[j];
                    NT t_aval = aval[j];
                    for (int k = brpt[t_acol]; k < brpt[t_acol + 1]; ++k) {
                        NT t_val = multop(t_aval, bval[k]);
	
                        int key = bcol[k];
                        int hash = (key * HASH_SCAL) & (ht_size - 1);
#ifdef VECTORIZE
                        key_m = _mm256_set1_epi32(key);
#endif
                        while (1) {
#ifdef VECTORIZE
                            check_m = _mm256_maskload_epi32(ht_check + (hash << VEC_LENGTH_BIT), true_m);
                            mask_m = _mm256_cmpeq_epi32(key_m, check_m);
                            mask = _mm256_movemask_epi8(mask_m);
                            if (mask != 0) {
                                int target = __builtin_ctz(mask) >> 2;
                                ht_value[(hash << VEC_LENGTH_BIT) + target] += t_val;
                                break;
                            }
#else
                            int flag = -1;
                            for (int l = 0; l < VEC_LENGTH; ++l) {
                                if (ht_check[(hash << VEC_LENGTH_BIT) + l] == key) {
                                    flag = l;
                                }
                            }
                            if (flag >= 0) {
                                ht_value[(hash << VEC_LENGTH_BIT) + flag] += t_val;
                                break;
                            }
#endif
                            else {
                                int cur_nz;
#ifdef VECTORIZE
                                mask_m = _mm256_cmpeq_epi32(check_m, init_m);
                                mask = _mm256_movemask_epi8(mask_m);
                                cur_nz = (32 - _popcnt32(mask)) >> 2;
#else
                                cur_nz = VEC_LENGTH;
                                for (int l = 0; l < VEC_LENGTH; ++l) {
                                    if (ht_check[(hash << VEC_LENGTH_BIT) + l] == -1) {
                                        cur_nz = l;
                                        break;
                                    }
                                }
#endif
                                if (cur_nz < VEC_LENGTH) {
                                    ht_check[(hash << VEC_LENGTH_BIT) + cur_nz] = key;
                                    ht_value[(hash << VEC_LENGTH_BIT) + cur_nz] = t_val;
                                    break;
                                }
                                else {
                                    hash = (hash + 1) & (ht_size - 1);
                                }
                            }
                        }
                    }
                }
                sort_and_store_table2mat<sortOutput, int, NT>(ht_check, ht_value,
                                                              ccol + offset, cval + offset,
                                                              crpt[i + 1] - offset, table_size);
            }
        }
    }
}

template <bool sortOutput, typename NT, typename MultiplyOperation, typename AddOperation>
inline void hash_numeric_vec(const long long int *arpt, const long long int *acol, const NT *aval, const long long int *brpt, const long long int *bcol, const NT *bval, const long long int *crpt, long long int *ccol, NT *cval, const BIN<long long int, NT> &bin, MultiplyOperation multop, AddOperation addop)
{
#ifdef VECTORIZE
    const __m256i init_m = _mm256_set1_epi64x(-1);
    const __m256i true_m = _mm256_set1_epi64x(0xffffffffffffffff);
#endif        

#pragma omp parallel
    {
        long long int tid = omp_get_thread_num();
        long long int start_row = bin.rows_offset[tid];
        long long int end_row = bin.rows_offset[tid + 1];

        long long int *ht_check = bin.local_hash_table_id[tid];
        NT *ht_value = bin.local_hash_table_val[tid];

        for (long long int i = start_row; i < end_row; ++i) {
#ifdef VECTORIZE
            __m256i key_m, check_m, mask_m;
            int mask;
#endif            
            long long int bid = bin.bin_id[i];

            if (bid > 0) {
                long long int offset = crpt[i];
                long long int table_size = MIN_HT_N << (bid - 1);
                long long int ht_size = table_size >> VEC_LENGTH_LONG_BIT;

                for (long long int j = 0; j < table_size; ++j) {
                    ht_check[j] = -1;
                }
  
                for (long long int j = arpt[i]; j < arpt[i + 1]; ++j) {
                    long long int t_acol = acol[j];
                    NT t_aval = aval[j];
                    for (long long int k = brpt[t_acol]; k < brpt[t_acol + 1]; ++k) {
                        NT t_val = multop(t_aval, bval[k]);
                        long long int key = bcol[k];
                        long long int hash = (key * HASH_SCAL) & (ht_size - 1);
#ifdef VECTORIZE
                        key_m = _mm256_set1_epi64x(key);
#endif
                        while (1) {
#ifdef VECTORIZE
                            check_m = _mm256_maskload_epi64(ht_check + (hash << VEC_LENGTH_LONG_BIT), true_m);
                            mask_m = _mm256_cmpeq_epi64(key_m, check_m);
                            mask = _mm256_movemask_epi8(mask_m);
                            if (mask != 0) {
                                int target = __builtin_ctz(mask) >> 3;
                                ht_value[(hash << VEC_LENGTH_LONG_BIT) + target] += t_val;
                                break;
                            }
#else
                            int flag = -1;
                            for (int l = 0; l < VEC_LENGTH_LONG; ++l) {
                                if (ht_check[(hash << VEC_LENGTH_LONG_BIT) + l] == key) {
                                    flag = l;
                                }
                            }
                            if (flag >= 0) {
                                ht_value[(hash << VEC_LENGTH_LONG_BIT) + flag] += t_val;
                                break;
                            }
#endif
                            else {
                                int cur_nz;
#ifdef VECTORIZE
                                mask_m = _mm256_cmpeq_epi64(check_m, init_m);
                                mask = _mm256_movemask_epi8(mask_m);
                                cur_nz = (32 - _popcnt32(mask)) >> 3;
#else
                                cur_nz = VEC_LENGTH_LONG;
                                for (int l = 0; l < VEC_LENGTH_LONG; ++l) {
                                    if (ht_check[(hash << VEC_LENGTH_LONG_BIT) + l] == -1) {
                                        cur_nz = l;
                                        break;
                                    }
                                }
#endif
                                if (cur_nz < VEC_LENGTH_LONG) {
                                    ht_check[(hash << VEC_LENGTH_LONG_BIT) + cur_nz] = key;
                                    ht_value[(hash << VEC_LENGTH_LONG_BIT) + cur_nz] = t_val;
                                    break;
                                }
                                else {
                                    hash = (hash + 1) & (ht_size - 1);
                                }
                            }
                        }
                    }
                }
                sort_and_store_table2mat<sortOutput, long long int, NT>(ht_check, ht_value,
                                                                        ccol + offset, cval + offset,
                                                                        crpt[i + 1] - offset, table_size);
            }
        }
    }
}
#endif

/*
 * Executing Hash SpGEMM
 * The function starts with initialization of hash table followed by symbolic phase and numeric phase with hash table.
 */
template <bool vectorProbing, bool sortOutput, typename IT, typename NT, typename MultiplyOperation, typename AddOperation>
void HashSpGEMM(const CSR<IT, NT> &a, const CSR<IT, NT> &b, CSR<IT, NT> &c, MultiplyOperation multop, AddOperation addop)
{
    BIN<IT, NT> bin(a.rows, MIN_HT_S);
  
    c.rows = a.rows;
    c.cols = b.cols;
    c.zerobased = true;

    /* Set max bin */
    bin.set_max_bin(a.rowptr, a.colids, b.rowptr, c.rows, c.cols);
    
    /* Create hash table (thread local) */
    bin.create_local_hash_table(c.cols);

    /* Symbolic Phase */
    c.rowptr = my_malloc<IT>(c.rows + 1);
    hash_symbolic<vectorProbing>(a.rowptr, a.colids, b.rowptr, b.colids, c.rowptr, bin, c.rows, &(c.nnz));
  
    c.colids = my_malloc<IT>(c.nnz);
    c.values = my_malloc<NT>(c.nnz);

    /* Numeric Phase */
    if (vectorProbing) {
        hash_numeric_vec<sortOutput>(a.rowptr, a.colids, a.values, b.rowptr, b.colids, b.values, c.rowptr, c.colids, c.values, bin, multop, addop);
    }
    else {
        hash_numeric<sortOutput>(a.rowptr, a.colids, a.values, b.rowptr, b.colids, b.values, c.rowptr, c.colids, c.values, bin, multop, addop);
    }
}

template <bool vectorProbing, bool sortOutput, typename IT, typename NT, typename MultiplyOperation, typename AddOperation>
void prof_HashSpGEMM(const CSR<IT, NT> &a, const CSR<IT, NT> &b, CSR<IT, NT> &c, MultiplyOperation multop, AddOperation addop)
{
    BIN<IT, NT> bin(a.rows, MIN_HT_S);
  
    c.rows = a.rows;
    c.cols = b.cols;
    c.zerobased = true;

    /* Set max bin */
    double t0;
    t0 = omp_get_wtime();
    bin.set_max_bin(a.rowptr, a.colids, b.rowptr, c.rows, c.cols);
    printf("set max bin time %le\n", omp_get_wtime() - t0);
    
    /* Create hash table (thread local) */
    t0 = omp_get_wtime();
    bin.create_local_hash_table(c.cols);
    printf("create_local_hash_table time %le\n", omp_get_wtime() - t0);

    /* Symbolic Phase */
    t0 = omp_get_wtime();
    c.rowptr = my_malloc<IT>(c.rows + 1);
    hash_symbolic<vectorProbing>(a.rowptr, a.colids, b.rowptr, b.colids, c.rowptr, bin, c.rows, &(c.nnz));
    printf("hash symbolic time %le\n", omp_get_wtime() - t0);
  
    t0 = omp_get_wtime();
    c.colids = my_malloc<IT>(c.nnz);
    c.values = my_malloc<NT>(c.nnz);

    /* Numeric Phase */
    if (vectorProbing) {
        hash_numeric_vec<sortOutput>(a.rowptr, a.colids, a.values, b.rowptr, b.colids, b.values, c.rowptr, c.colids, c.values, bin, multop, addop);
    }
    else {
        hash_numeric<sortOutput>(a.rowptr, a.colids, a.values, b.rowptr, b.colids, b.values, c.rowptr, c.colids, c.values, bin, multop, addop);
    }
    printf("hash numeric time %le\n", omp_get_wtime() - t0);
}

/*
 * Hash SpGEMM functions called without full template values
 */
template <bool sortOutput, typename IT, typename NT, typename MultiplyOperation, typename AddOperation>
void HashSpGEMM(const CSR<IT, NT> &a, const CSR<IT, NT> &b, CSR<IT, NT> &c, MultiplyOperation multop, AddOperation addop)
{
    HashSpGEMM<false, sortOutput, IT, NT, MultiplyOperation, AddOperation>(a, b, c, multop, addop);
}

template <typename IT, typename NT, typename MultiplyOperation, typename AddOperation>
void HashSpGEMM(const CSR<IT, NT> &a, const CSR<IT, NT> &b, CSR<IT, NT> &c, MultiplyOperation multop, AddOperation addop)
{
    HashSpGEMM<false, true, IT, NT, MultiplyOperation, AddOperation>(a, b, c, multop, addop);
}

#endif
