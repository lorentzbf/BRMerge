#ifndef __Z_TIMING_H__
#define __Z_TIMING_H__

class Upper_Timing {
    public:
    bool measure_separate;
    bool measure_whole;
    double thread;
    double pre_allocate;
    double compute_flop;
    double prefix_sum_flop;
    double load_balance;
    double allocate_hook;
    double compute;
    double prefix_sum_nnz;
    double allocate_C;
    double copy;
    double cleanup;
    double total;
    Upper_Timing();
    void operator+=(const Upper_Timing& a);
    void operator/=(const double n);
    void print(const double total_flop);
    void reg_print(const double total_flop);
};

class Precise_Timing {
    public:
    bool measure_separate;
    bool measure_whole;
    double pre_allocate;
    double compute_flop;
    double prefix_sum_flop;
    double load_balance;
    double symbolic;
    double prefix_sum_nnz;
    double allocate_C;
    double compute;
    double cleanup;
    double total;
    Precise_Timing();
    void operator+=(const Precise_Timing& a);
    void operator/=(const double n);
    void print(const double total_flop);
    void reg_print(const double total_flop);
};

#endif


