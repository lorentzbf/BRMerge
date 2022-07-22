#ifndef _Z_UTILS_H_
#define _Z_UTILS_H_

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

#endif
