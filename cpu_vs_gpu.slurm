#!/bin/bash -l
#SBATCH --cluster=wice
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --account=lp_h_pds_iiw
#SBATCH --time=00:10:00
#SBATCH --output=%x.o%A
#SBATCH --error=%x.e%A

source /data/leuven/303/vsc30380/slurmhooks

module purge
module load CUDA/11.7.0
module load GCC/10.3.0

echo "============================="
echo "Compiling CPU + GPU versions"
echo "============================="

mkdir -p compiled

g++ -O3 -o compiled/cpu_serial CPU_serial.cpp -lm

nvcc -O3 -DNDEBUG \
    -o compiled/gpu_optimal GPU_optimal.cu \
    -gencode arch=compute_80,code=sm_80 \
    -gencode arch=compute_80,code=compute_80 \
    -diag-suppress 550 \
    -I.

nvcc -O3 -DNDEBUG \
    -o compiled/gpu_optimal_precompute GPU_optimal_precompute.cu \
    -gencode arch=compute_80,code=sm_80 \
    -gencode arch=compute_80,code=compute_80 \
    -diag-suppress 550 \
    -I.

echo "============================="
echo "Running tests"
echo "============================="

INPUT=img/in/4k.jpg
OUT_CPU=img/out/cpu_out.png
OUT_GPU=img/out/gpu_out.png

echo "--- CPU Serial ---"
./compiled/cpu_serial $INPUT $OUT_CPU
echo "--- GPU Optimal ---"
./compiled/gpu_optimal $INPUT $OUT_GPU
echo "--- GPU Optimal Precompute ---"
./compiled/gpu_optimal_precompute $INPUT $OUT_GPU

echo "DONE"