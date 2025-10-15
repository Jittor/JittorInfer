#!/bin/bash

cd build

export OMPI_ALLOW_RUN_AS_ROOT=1
export  OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
make llama-test-mpi-dp-ep -j32
mpirun -n 8 ./bin/llama-test-mpi-dp-ep
