#include "all.h"
#include "mkl_mult.h"
#include "Timings.h"

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
	
    CSR<int, double> A, B;
    A.construct(mat1_file);
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

    CSR<int, double> C;
    omp_set_num_threads(64);
    
    long total_flop = compute_flop(A, B);
    double total_flop_G = double(total_flop * 2) / 1000000000;

    Timings timing, benchtiming;
    mkl(A, B, C, timing);
    timing.print(total_flop * 2);
    C.make_empty();


    int iter = 10;
    for(int i = 0; i < iter; i++){
        mkl(A, B, C, timing);
        benchtiming += timing;
        if(i < iter - 1){
            C.make_empty();
        }
    }
    benchtiming /= iter;
    benchtiming.print(total_flop * 2);
    C.make_empty();
    A.make_empty();
    B.make_empty();

}

