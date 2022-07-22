#include "Timings.h"
#include <stdio.h>

Timings::Timings(){
    measure_separate = true;
    measure_total = true;
    create = 0;
    spmm = 0;
    convert = 0;
    order = 0;
    export_csr = 0;
    destroy = 0;
    total = 0;
}

void Timings::operator+=(const Timings& a){
    create += a.create;
    spmm += a.spmm;
    convert += a.convert;
    order += a.order;
    export_csr += a.export_csr;
    destroy += a.destroy;
    total += a.total;
}

void Timings::operator/=(const double n){
    create /= n;
    spmm /= n;
    convert /= n;
    order /= n;
    export_csr /= n;
    destroy /= n;
    total /= n;
}

void Timings::print(double total_flop){
    double total_flop_G = total_flop/1000000000;
    printf("total flop %lf\n", total_flop);
    double sum_total = create + spmm + convert + order + export_csr + destroy;
    if(measure_separate){
        printf("time(ms):\n");
        printf("    create             %8.3lfms %6.2lf%%\n", 1000*create, create/total*100);
        printf("    spmm               %8.3lfms %6.2lf%%\n", 1000*spmm, spmm/total*100);
        printf("    convert            %8.3lfms %6.2lf%%\n", 1000*convert, convert/total*100);
        printf("    order              %8.3lfms %6.2lf%%\n", 1000*order, order/total*100);
        printf("    export_csr         %8.3lfms %6.2lf%%\n", 1000*export_csr, export_csr/total*100);
        printf("    destroy            %8.3lfms %6.2lf%%\n", 1000*destroy, destroy/total*100);
        printf("    sum_total          %8.3lfms %6.2lf%%\n", 1000*sum_total, sum_total/total*100);
        //printf("\e[1;31m    symbolic_binning %8.3lfms %6.2lf%%\n\e[0m", 1000*symbolic_binning, symbolic_binning/total*100);

        printf("perf(Gflops):\n");
        printf("    create             %6.2lf\n", total_flop_G/create);
        printf("    spmm               %6.2lf\n", total_flop_G/spmm);
        printf("    convert            %6.2lf\n", total_flop_G/convert);
        printf("    order              %6.2lf\n", total_flop_G/order);
        printf("    export_csr         %6.2lf\n", total_flop_G/export_csr);
        printf("    destroy            %6.2lf\n", total_flop_G/destroy);
        printf("    total              %6.2lf\n", total_flop_G/total);
    }
}
        
void Timings::reg_print(double total_flop){
    double total_flop_G = total_flop/1000000000;
    printf("%le\n", total_flop_G/ total);
}
