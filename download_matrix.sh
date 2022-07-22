#!/bin/bash

if [ -d matrix/suite_sparse ]; then
    cd matrix/suite_sparse
else
    mkdir -p matrix/suite_sparse
    cd matrix/suite_sparse
fi

# download patents_main
if [ ! -e patents_main ]; then
    wget https://suitesparse-collection-website.herokuapp.com/MM/Pajek/patents_main.tar.gz
    tar zxvf patents_main.tar.gz
fi
echo Successfully downloaded the matrix.

