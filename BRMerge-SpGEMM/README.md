# Get started
1 Compile the code: 
- Make sure Intel's oneAPI base toolkit and HPC toolkit are installed. 
- Open Makefile and correct the installation path in the INCLUDE variable. 
- Execute ``` $> bash make.sh ``` to compile all the approaches.

2 Execute the approaches by ``` $> cmd matrix1_name [matrix2_name] [num_threads]```.
For example, ``` $> ./reg_brmerge_precise patents_main patents_main 80``` computes **patents_main** matrix multiplied by **patents_main** matrix with **80** CPU threads.
The **reg\_** prefix means the regression version of the approaches where only the matrix name and GFLOPS results are printed.


