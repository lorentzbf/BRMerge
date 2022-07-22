#ifndef _Z_UTILS_H_
#define _Z_UTILS_H_
#include <scalable_allocator.h>

namespace util{

template <typename T>
T sum(T *a, int n){
    T res = 0;
    for(int i = 0; i < n; i++){
        res += a[i];
    }
    return res;
}

template <typename T>
T mean(T *a, int n){
    return sum(a,n)/n;
}

template <typename T>
T max(T *a, int n){
    T res = a[0];
    for(int i = 1; i < n; i++){
        if(res < a[i]){
            res = a[i];
        }
    }
    return res;
}

template <typename T>
T min(T *a, int n){
    T res = a[0];
    for(int i = 1; i < n; i++){
        if(res > a[i]){
            res = a[i];
        }
    }
    return res;
}

}

template <typename T>
inline T* my_malloc(unsigned long array_size)
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

#endif
