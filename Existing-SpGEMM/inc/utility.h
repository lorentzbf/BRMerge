#ifndef YUSUKE_UTILITY_H
#define YUSUKE_UTILITY_H

#include <stdlib.h>
#include <stdint.h>
#include <climits>
#include <iostream>
#include <cmath>
#include <vector>
#include <fstream>
#include <scalable_allocator.h>
#include <omp.h>
//#include <tbb/scalable_allocator.h>

using namespace std;
#define 	EPSILON   0.001


template <class T>
struct ErrorTolerantEqual:
public binary_function< T, T, bool >
{
   ErrorTolerantEqual(const T & myepsilon):epsilon(myepsilon) {};
   inline bool operator() (const T & a, const T & b) const
   {
   	// According to the IEEE 754 standard, negative zero and positive zero should 
   	// compare as equal with the usual (numerical) comparison operators, like the == operators of C++ 
   
  	if(a == b)      // covers the "division by zero" case as well: max(a,b) can't be zero if it fails
   		return true;    // covered the integral numbers case
   
   	return ( std::abs(a - b) < epsilon || (std::abs(a - b) / max(std::abs(a), std::abs(b))) < epsilon ) ; 
   }
   T epsilon;
};

// Because identify reports ambiguity in PGI compilers
template<typename T>
struct myidentity : public std::unary_function<T, T>
{
    const T operator()(const T& x) const
    {
        return x;
    }
};

template<typename _ForwardIterator, typename _StrictWeakOrdering>
bool my_is_sorted(_ForwardIterator __first, _ForwardIterator __last,  _StrictWeakOrdering __comp)
{
   if (__first == __last)
   	return true;
   
   _ForwardIterator __next = __first;
   for (++__next; __next != __last; __first = __next, ++__next)
   	if (__comp(*__next, *__first))
   		return false;
   	return true;
};

template <typename ITYPE>
ITYPE CumulativeSum (ITYPE * arr, ITYPE size)
{
    ITYPE prev;
    ITYPE tempnz = 0 ;
    for (ITYPE i = 0 ; i < size ; ++i)
    {
		prev = arr[i];  
		arr[i] = tempnz; 
		tempnz += prev ; 
    }
    return (tempnz) ;		    // return sum
}


template<typename _ForwardIter, typename T>
void iota(_ForwardIter __first, _ForwardIter __last, T __value)
{
	while (__first != __last)
     		*__first++ = __value++;
}
	
template<typename T, typename I>
T ** allocate2D(I m, I n)
{
	T ** array = new T*[m];
	for(I i = 0; i<m; ++i) 
		array[i] = new T[n];
	return array;
}

template<typename T, typename I>
void deallocate2D(T ** array, I m)
{
	for(I i = 0; i<m; ++i) 
		delete [] array[i];
	delete [] array;
}


template < typename T >
struct absdiff : binary_function<T, T, T>
{
        T operator () ( T const &arg1, T const &arg2 ) const
        {
                using std::abs;
                return abs( arg1 - arg2 );
        }
};

/* This function will return n % d.
   d must be one of: 1, 2, 4, 8, 16, 32, … */
inline unsigned int getModulo(unsigned int n, unsigned int d)
{
	return ( n & (d-1) );
} 

// Same requirement (d=2^k) here as well
inline unsigned int getDivident(unsigned int n, unsigned int d)
{
	while((d = d >> 1))
		n = n >> 1;
	return n;
}

// Memory allocation by C++-new / Aligned malloc / scalable malloc
template <typename T>
inline T* my_malloc(size_t array_size)
{
#ifdef CPP
    return (T *)(new T[array_size]);
#elif defined IMM
    return (T *)_mm_malloc(sizeof(T) * array_size, 64);
#elif defined TBB
    return (T *)scalable_malloc(sizeof(T) * array_size);
#else
    return (T *)scalable_malloc(sizeof(T) * array_size);
#endif
}

// Memory deallocation
template <typename T>
inline void my_free(T *a)
{
#ifdef CPP
    delete[] a;
#elif defined IMM
    _mm_free(a);
#elif defined TBB
    scalable_free(a);
#else
    scalable_free(a);
#endif
}

// Prefix sum (Sequential)
template <typename T1, typename T2>
void seq_scan(T1 *in, T2 *out, int N)
{
    out[0] = 0;
    for (int i = 0; i < N - 1; ++i) {
        out[i + 1] = out[i] + in[i];
    }
}

// Prefix sum (Thread parallel)
template <typename T1, typename T2>
void scan(T1 *in, T2 *out, int N)
{
    if (N < (1 << 17)) {
        seq_scan(in, out, N);
    }
    else {
        int tnum;
        // my modify, different from yusuke, not much difference in performance
        #pragma omp parallel
        {
            #pragma omp single
            {
                tnum = omp_get_num_threads();
            }
        }
        int each_n = N / tnum;
        //T2 *partial_sum = (T *)scalable_malloc(sizeof(T) * (tnum));
        T2 *partial_sum = new T2 [tnum];
#pragma omp parallel num_threads(tnum)
        {
            int tid = omp_get_thread_num();
            int start = each_n * tid;
            int end = (tid < tnum - 1)? start + each_n : N;
            out[start] = 0;
            for (int i = start; i < end - 1; ++i) {
                out[i + 1] = out[i] + in[i];
            }
            partial_sum[tid] = out[end - 1] + in[end - 1];
#pragma omp barrier

            int offset = 0;
            for (int i = 0; i < tid; ++i) {
                offset += partial_sum[i];
            }
            for (int i = start; i < end; ++i) {
                out[i] += offset;
            }
        }
        //out[N] = out[N-1] + in[N-1];
        //scalable_free(partial_sum);
        delete [] partial_sum;
    }
}

template <typename T1, typename T2>
void seq_prefix_sum(T1 *in, T2 *out, int N)
{
    out[0] = 0;
    for (int i = 0; i < N; ++i) {
        out[i + 1] = out[i] + in[i];
    }
}

template <typename T1, typename T2>
void para_prefix_sum(T1 *in, T2 *out, int N, int tnum){

    //printf("num threads %d\n", tnum);
    int each_n = N / tnum;
    //T2 *partial_sum = (T *)scalable_malloc(sizeof(T) * (tnum));
    T2 *partial_sum = new T2 [tnum];
#pragma omp parallel num_threads(tnum)
    {
        int tid = omp_get_thread_num();
        int start = each_n * tid;
        int end = (tid < tnum - 1)? start + each_n : N;
        out[start] = 0;
        for (int i = start; i < end - 1; ++i) {
            out[i + 1] = out[i] + in[i];
        }
        partial_sum[tid] = out[end - 1] + in[end - 1];
#pragma omp barrier
        int offset = 0;
        for (int i = 0; i < tid; ++i) {
            offset += partial_sum[i];
        }
        for (int i = start; i < end; ++i) {
            out[i] += offset;
        }
        if(tid == tnum - 1)
            out[N] = out[N-1] + in[N-1];
    }
    //scalable_free(partial_sum);
    delete [] partial_sum;
}

template <typename T1, typename T2>
inline void opt_prefix_sum(T1 *in, T2 *out, int N, int tnum = 64){
    if( N < 1 << 13){
        seq_prefix_sum(in, out, N);
    }
    else{
        para_prefix_sum(in, out, N, tnum);
    }
}

template <typename T>
int find_approx(T *arr, int N, T elem){
    int s = 0;
    int e = N-1;
    assert(elem >= arr[0] && "find approx elem < smallest");
    if(elem >= arr[N-1]){
        return N-1;
    }
    while(true){
        int m = (s+e)/2;
        if(arr[m] == elem){
            return m;
        }
        else if(arr[m] < elem){
            if(elem < arr[m+1]){
                return m;
            }
            s = m;
        }
        else{ // elem < arr[m]
            if(arr[m-1] < elem){
                return m-1;
            }
            e = m;
        }
    }
}

// Sort by key
template <typename IT, typename NT>
inline void mergesort(IT *nnz_num, NT *nnz_sorting,
               IT *temp_num, NT *temp_sorting,
               IT left, IT right)
{
    int mid, i, j, k;
  
    if (left >= right) {
        return;
    }

    mid = (left + right) / 2;

    mergesort(nnz_num, nnz_sorting, temp_num, temp_sorting, left, mid);
    mergesort(nnz_num, nnz_sorting, temp_num, temp_sorting, mid + 1, right);

    for (i = left; i <= mid; ++i) {
        temp_num[i] = nnz_num[i];
        temp_sorting[i] = nnz_sorting[i];
    }

    for (i = mid + 1, j = right; i <= right; ++i, --j) {
        temp_sorting[i] = nnz_sorting[j];
        temp_num[i] = nnz_num[j];
    }

    i = left;
    j = right;
  
    for (k = left; k <= right; ++k) {
        if (temp_num[i] <= temp_num[j] && i <= mid) {
            nnz_num[k] = temp_num[i];
            nnz_sorting[k] = temp_sorting[i++];
        }
        else {
            nnz_num[k] = temp_num[j];
            nnz_sorting[k] = temp_sorting[j--];
        }
    }
}

/*Sorting key-value*/
template <typename IT, typename NT>
inline void cpu_sorting_key_value(IT *key, NT *value, IT N)
{
    IT *temp_key;
    NT *temp_value;

    temp_key = my_malloc<IT>(N);
    temp_value = my_malloc<NT>(N);

    mergesort(key, value, temp_key, temp_value, 0, N-1);

    my_free<IT>(temp_key);
    my_free<NT>(temp_value);

}

template <class index_type, class value_type>
class Pair{
    public:
    index_type ind;
    value_type val;
    friend bool operator<=(const Pair &lhs, const Pair& rhs){
        return lhs.ind <= rhs.ind;
    }
    friend bool operator<(const Pair &lhs, const Pair& rhs){
        return lhs.ind < rhs.ind;
    }
    friend bool operator>(const Pair &lhs, const Pair& rhs){
        return lhs.ind > rhs.ind;
    }
};

#endif

