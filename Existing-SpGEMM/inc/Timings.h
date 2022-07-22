#ifndef __Z_TIMING_H__
#define __Z_TIMING_H__

class Timings {
    public:
    bool measure_separate;
    bool measure_total;
    double create;
    double spmm;
    double convert;
    double order;
    double export_csr;
    double destroy;
    double total;

    Timings();
    void operator+=(const Timings& b);
    void operator/=(const double x);
    void print(const double total_flop);
    void reg_print(const double total_flop);
};

#endif


