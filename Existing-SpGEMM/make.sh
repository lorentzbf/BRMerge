#!/bin/bash

make heap_spgemm
make hash_spgemm
make hashvec_spgemm
make outer_spgemm
make mkl_spgemm

make reg_heap_spgemm
make reg_hash_spgemm
make reg_hashvec_spgemm
make reg_outer_spgemm
make reg_mkl_spgemm
