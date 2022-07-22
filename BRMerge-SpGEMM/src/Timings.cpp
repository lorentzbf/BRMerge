#include "Timings.h"
#include <stdio.h>

Upper_Timing::Upper_Timing(){
    measure_separate = true;
    measure_whole = true;
    thread = 0;
    pre_allocate = 0;
    compute_flop = 0;
    prefix_sum_flop = 0;
    load_balance = 0;
    allocate_hook = 0;
    compute = 0;
    prefix_sum_nnz = 0;
    allocate_C = 0;
    copy = 0;
    cleanup = 0;
    total = 0;
}

void Upper_Timing::operator+=(const Upper_Timing& a){
    thread += a.thread;
    pre_allocate += a.pre_allocate;
    compute_flop += a.compute_flop;
    prefix_sum_flop += a.prefix_sum_flop;
    load_balance += a.load_balance;
    allocate_hook += a.allocate_hook;
    compute += a.compute;
    prefix_sum_nnz += a.prefix_sum_nnz;
    allocate_C += a.allocate_C;
    copy += a.copy;
    cleanup += a.cleanup;
    total += a.total;
}

void Upper_Timing::operator/=(const double n){
    thread /= n;
    pre_allocate /= n;
    compute_flop /= n;
    prefix_sum_flop /= n;
    load_balance /= n;
    allocate_hook /= n;
    compute /= n;
    prefix_sum_nnz /= n;
    allocate_C /= n;
    copy /= n;
    cleanup /= n;
    total /= n;
}

void Upper_Timing::print(double total_flop){
    double total_flop_G = total_flop/1000000000;
    printf("total flop %lf\n", total_flop);
    double sum_total =  thread + pre_allocate + compute_flop + prefix_sum_flop + load_balance + allocate_hook + compute + prefix_sum_nnz + allocate_C + copy + cleanup;
    if(measure_separate){
        printf("time(ms):\n");
        printf("    thread              %8.3lfms %6.2lf%%\n", 1000*thread, thread/total*100);
        printf("    pre_allocate        %8.3lfms %6.2lf%%\n", 1000*pre_allocate, pre_allocate/total*100);
        printf("    compute_flop        %8.3lfms %6.2lf%%\n", 1000*compute_flop, compute_flop/total*100);
        printf("    prefix_sum_flop     %8.3lfms %6.2lf%%\n", 1000*prefix_sum_flop, prefix_sum_flop/total*100);
        printf("    load_balance        %8.3lfms %6.2lf%%\n", 1000*load_balance, load_balance/total*100);
        printf("    allocate_hook       %8.3lfms %6.2lf%%\n", 1000*allocate_hook, allocate_hook/total*100);
        printf("    compute             %8.3lfms %6.2lf%%\n", 1000*compute, compute/total*100);
        printf("    prefix_sum_nnz      %8.3lfms %6.2lf%%\n", 1000*prefix_sum_nnz, prefix_sum_nnz/total*100);
        printf("    allocate_C          %8.3lfms %6.2lf%%\n", 1000*allocate_C, allocate_C/total*100);
        printf("    copy                %8.3lfms %6.2lf%%\n", 1000*copy, copy/total*100);
        printf("    cleanup             %8.3lfms %6.2lf%%\n", 1000*cleanup, cleanup/total*100);
        printf("    sum_total           %8.3lfms %6.2lf%%\n", 1000*sum_total, sum_total/total*100);
        printf("perf(GFLOPS):\n");
        printf("    thread              %6.2lf\n", total_flop_G/thread);
        printf("    pre_allocate        %6.2lf\n", total_flop_G/pre_allocate);
        printf("    compute_flop        %6.2lf\n", total_flop_G/compute_flop);
        printf("    prefix_sum_flop     %6.2lf\n", total_flop_G/prefix_sum_flop);
        printf("    load_balance        %6.2lf\n", total_flop_G/load_balance);
        printf("    allocate_hook       %6.2lf\n", total_flop_G/allocate_hook);
        printf("    compute             %6.2lf\n", total_flop_G/compute);
        printf("    prefix_sum_nnz      %6.2lf\n", total_flop_G/prefix_sum_nnz);
        printf("    allocate_C          %6.2lf\n", total_flop_G/allocate_C);
        printf("    copy                %6.2lf\n", total_flop_G/copy);
        printf("    cleanup             %6.2lf\n", total_flop_G/cleanup);
        printf("    total               %6.2lf\n", total_flop_G/total);
    }
}
void Upper_Timing::reg_print(double total_flop){
    double total_flop_G = total_flop/1000000000;
    printf("%le\n", total_flop_G/total);
}


// Precise Timing
Precise_Timing::Precise_Timing(){
    measure_separate = true;
    measure_whole = true;
    pre_allocate = 0;
    compute_flop = 0;
    prefix_sum_flop = 0;
    load_balance = 0;
    symbolic = 0;
    prefix_sum_nnz = 0;
    allocate_C = 0;
    compute = 0;
    cleanup = 0;
    total = 0;
}

void Precise_Timing::operator+=(const Precise_Timing& a){
    pre_allocate += a.pre_allocate;
    compute_flop += a.compute_flop;
    prefix_sum_flop += a.prefix_sum_flop;
    load_balance += a.load_balance;
    symbolic += a.symbolic;
    prefix_sum_nnz += a.prefix_sum_nnz;
    allocate_C += a.allocate_C;
    compute += a.compute;
    cleanup += a.cleanup;
    total += a.total;
}

void Precise_Timing::operator/=(const double n){
    pre_allocate /= n;
    compute_flop /= n;
    prefix_sum_flop /= n;
    load_balance /= n;
    symbolic /= n;
    prefix_sum_nnz /= n;
    allocate_C /= n;
    compute /= n;
    cleanup /= n;
    total /= n;
}

void Precise_Timing::print(double total_flop){
    double total_flop_G = total_flop/1000000000;
    printf("total flop %lf\n", total_flop);
    double sum_total =  pre_allocate + compute_flop + prefix_sum_flop + load_balance + symbolic + prefix_sum_nnz + allocate_C + compute;
    if(measure_separate){
        printf("time(ms):\n");
        printf("    pre_allocate        %8.3lfms %6.2lf%%\n", 1000*pre_allocate, pre_allocate/total*100);
        printf("    compute_flop        %8.3lfms %6.2lf%%\n", 1000*compute_flop, compute_flop/total*100);
        printf("    prefix_sum_flop     %8.3lfms %6.2lf%%\n", 1000*prefix_sum_flop, prefix_sum_flop/total*100);
        printf("    load_balance        %8.3lfms %6.2lf%%\n", 1000*load_balance, load_balance/total*100);
        printf("    symbolic            %8.3lfms %6.2lf%%\n", 1000*symbolic, symbolic/total*100);
        printf("    prefix_sum_nnz      %8.3lfms %6.2lf%%\n", 1000*prefix_sum_nnz, prefix_sum_nnz/total*100);
        printf("    allocate_C          %8.3lfms %6.2lf%%\n", 1000*allocate_C, allocate_C/total*100);
        printf("    compute             %8.3lfms %6.2lf%%\n", 1000*compute, compute/total*100);
        printf("    cleanup             %8.3lfms %6.2lf%%\n", 1000*cleanup, cleanup/total*100);
        printf("    sum_total           %8.3lfms %6.2lf%%\n", 1000*sum_total, sum_total/total*100);
        printf("perf(GFLOPS):\n");
        printf("    pre_allocate        %6.2lf\n", total_flop_G/pre_allocate);
        printf("    compute_flop        %6.2lf\n", total_flop_G/compute_flop);
        printf("    prefix_sum_flop     %6.2lf\n", total_flop_G/prefix_sum_flop);
        printf("    load_balance        %6.2lf\n", total_flop_G/load_balance);
        printf("    symbolic            %6.2lf\n", total_flop_G/symbolic);
        printf("    prefix_sum_nnz      %6.2lf\n", total_flop_G/prefix_sum_nnz);
        printf("    allocate_C          %6.2lf\n", total_flop_G/allocate_C);
        printf("    compute             %6.2lf\n", total_flop_G/compute);
        printf("    cleanup             %6.2lf\n", total_flop_G/cleanup);
        printf("    total               %6.2lf\n", total_flop_G/total);
    }
}

void Precise_Timing::reg_print(double total_flop){
    double total_flop_G = total_flop/1000000000;
    printf("%lf\n", total_flop_G/total);
}

