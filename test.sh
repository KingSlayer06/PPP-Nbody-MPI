#!/bin/bash

echo "Testing correctness with 32 bodies..."
mpiexecjl -np 1 julia -O3 nbody-par-run.jl 32 1000 1 > par-1.txt
julia -O3 nbody-seq-run.jl 32 1000 1 > seq-1.txt
echo "Differences (should be none or minimal):"
diff seq-1.txt par-1.txt | head -20

echo -e "\nTesting with 100 bodies..."
echo "Sequential:"
julia -O3 nbody-seq-run.jl -s 500 10000 1
echo -e "\nParallel (2 processes):"
mpiexecjl -np 2 julia -O3 nbody-par-run.jl -s 500 10000 1
echo -e "\nParallel (4 processes):"
mpiexecjl -np 4 julia -O3 nbody-par-run.jl -s 500 10000 1
echo -e "\nParallel (8 processes):"
mpiexecjl -np 8 julia -O3 nbody-par-run.jl -s 500 10000 1