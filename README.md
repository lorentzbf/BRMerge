This repository contains the source code of BRMerge-Precise, BRMerge-Upper, Heap-SpGEMM, Hash-SpGEMM, Hashvec-SpGEMM, PB-SpGEMM(i.e., outer\_spgemm), and MKL-SpGEMM.
# Tested evironment
- Compiler: Intel ICC 2021.5.0 20211109
- CPU:
  - Intel Xeon Platinum 8163
  - Intel Xeon Gold 6254 
- Operating system: Ubuntu 18.04 LTS

# Get started
1 Execute ```$> bash download_matrix.sh``` in the current directory to download the matrix patents_main into matrix/suite_sparse directory

2 For detailed execution instruction, refer the ```README.md``` in the BRMerge-SpGEMM, and Existing-SpGEMM sub-directory.
