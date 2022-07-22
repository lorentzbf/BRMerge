#ifndef _Z_COMMON_
#define _Z_COMMON_

#include "define.h"
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <cassert>
#include <cstring>
#include "Timings.h"
#include "utils.h"


#define HP_TIMING_NOW(Var) \
    ({ unsigned int _hi, _lo; \
    asm volatile ("lfence\n\trdtsc" : "=a" (_lo), "=d" (_hi)); \
    (Var) = ((unsigned long long int) _hi << 32) | _lo; })

inline void cpuid(int *info, int eax, int ecx = 0){
    int ax, bx, cx, dx;
    __asm__ __volatile__ ("cpuid": "=a" (ax), "=b" (bx), "=c" (cx), "=d" (dx) : "a" (eax));

    info[0] = ax;
    info[1] = bx;
    info[2] = cx;
    info[3] = dx;
}

inline long get_tsc_freq(){
    static long freq = 0;
    if(unlikely((freq == 0))){
        int raw[4];
        cpuid(raw, 0x16); // get cpu freq
        freq = long(raw[0]) * 1000000;
        //printf("static first call %ld\n", freq);
    }
    return freq;
}

inline double fast_clock_time(){
    long counter;
    HP_TIMING_NOW(counter);
    return double(counter)/get_tsc_freq();
}

#endif
