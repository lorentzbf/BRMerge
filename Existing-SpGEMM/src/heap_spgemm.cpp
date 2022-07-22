#include "all.h"
#include "mkl_mult.h"

int main(int argc, char **argv)
{
    std::string mat1, mat2;
    mat1 = "can_24";
    mat2 = "can_24";
    if(argc == 2){
        mat1 = argv[1];
        mat2 = argv[1];
    }
    if(argc >= 3){
        mat1 = argv[1];
        mat2 = argv[2];
    }
    std::string mat1_file;
    if(mat1.find("ER") != std::string::npos){
        mat1_file = "../matrix/ER/" + mat1 +".mtx";
    }
    else if(mat1.find("G500") != std::string::npos){
        mat1_file = "../matrix/G500/" + mat1 +".mtx";
    }
    else{
        mat1_file = "../matrix/suite_sparse/" + mat1 + "/" + mat1 +".mtx";
    }
    std::string mat2_file;
    if(mat2.find("ER") != std::string::npos){
        mat2_file = "../matrix/ER/" + mat2 +".mtx";
    }
    else if(mat2.find("G500") != std::string::npos){
        mat2_file = "../matrix/G500/" + mat2 +".mtx";
    }
    else{
        mat2_file = "../matrix/suite_sparse/" + mat2 + "/" + mat2 +".mtx";
    }
    
    double t0, t1;
    t0 = fast_clock_time();	
    CSR<int, double> A, B;
    A.construct(mat1_file);
    t1 = fast_clock_time() - t0;
    printf("read file time %le\n", t1);

    if(mat1 == mat2){
        B = A;
    }
    else{
        B.construct(mat2_file);
        if(A.cols == B.rows){
            // do nothing
        }
        else if(A.cols < B.rows){
            CSR<int, double> tmp(B, A.cols, B.cols, 0, 0);
            B = tmp;
        }
        else{
            CSR<int, double> tmp(A, A.rows, B.rows, 0, 0);
            A = tmp;
        }
    }

    CSC<int, double> A_csc, B_csc, C_csc;
    t0 = fast_clock_time();
    convert(A_csc, A);
    convert(B_csc, B);
    t1 = fast_clock_time() - t0;
    printf("convert time %le s\n", t1);
    
    int tnum = 64;
    omp_set_num_threads(tnum);
    long flop = get_flop(A_csc, B_csc);
    double flop_G = (double)flop * 2 / 1000000000;
    printf("twice flop %ld\n", flop*2);
    
    t0 = fast_clock_time();
    HeapSpGEMM(A_csc, B_csc, C_csc, std::multiplies<double>(), plus<double>());
    t1 = fast_clock_time() - t0;
    printf("warmup time %le s, %le GFLOPS\n", t1, flop_G/t1);
    C_csc.make_empty();

    int iter = 1;
    t1 = 0;
    for (int i = 0; i < iter; ++i) {
        t0 = fast_clock_time();
        HeapSpGEMM(A_csc, B_csc, C_csc, std::multiplies<double>(), plus<double>());
        t1 += fast_clock_time() - t0;
        if (i < iter - 1) {
            C_csc.make_empty();
        }
    }
    t1 /= iter;
    printf("spgemm time %le s, %le GFLOPS\n", t1, flop_G/t1);
   

    // compare result
    CSR<int, double> C;
    convert(C, C_csc);
    CSR<int, double> C_ref;
    mkl(A, B, C_ref);
    if(C == C_ref){
        printf("pass\n");
    }
    else{
        printf("fail\n");
    }

    //C_csc.make_empty();

}

