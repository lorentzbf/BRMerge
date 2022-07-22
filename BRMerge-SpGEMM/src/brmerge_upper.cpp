#include "spgemm.h"
#include "mkl_spgemm.h"

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
	
    CSR A, B;
    A.construct(mat1_file);
    if(mat1 == mat2){
        B = A;
    }
    else{
        B.construct(mat2_file);
        if(A.N == B.M){
            // do nothing
        }
        else if(A.N < B.M){
            CSR tmp(B, A.N, B.N, 0, 0);
            B = tmp;
        }
        else{
            CSR tmp(A, A.M, B.M, 0, 0);
            A = tmp;
        }
    }

    CSR C;
    int num_threads = 64;
    if(argc >= 4){
        num_threads = atoi(argv[3]);
    }
    omp_set_num_threads(num_threads);

    long total_flop = compute_flop(A, B);
    printf("twice flop %ld\n", total_flop*2);

    Upper_Timing timing, benchtiming;
    brmerge_upper(A, B, C, timing);
    timing.print(total_flop * 2);
    C.release();


    int iter = 10;
    for(int i = 0; i < iter; i++){
        brmerge_upper(A, B, C, timing);
        benchtiming += timing;
        if(i < iter - 1){
            C.release();
        }
    }
    benchtiming /= iter;
    benchtiming.print(total_flop * 2);


    // compare result
    CSR C_ref;
    mkl(A, B, C_ref);
    if(C == C_ref){
        printf("pass\n");
    }
    else{
        printf("fail\n");
    }

    C.release();
}

