#!/bin/bash -l
#SBATCH --cluster=wice
#SBATCH --partition=a100
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

echo "Compiling CUDA kernels..."

# Create output directory
mkdir -p compiled

# ======================
# Compile unoptimized version
# ======================
nvcc -O3 -DNDEBUG \
    -o compiled/gaussian_param_unop GPU_param_unoptimized.cu \
    -gencode arch=compute_80,code=sm_80 \
    -gencode arch=compute_80,code=compute_80 \
    -diag-suppress 550 \
    -I.

# ======================
# Compile precomputed version
# ======================
nvcc -O3 -DNDEBUG \
    -o compiled/gaussian_param_precomp GPU_param_precompute.cu \
    -gencode arch=compute_80,code=sm_80 \
    -gencode arch=compute_80,code=compute_80 \
    -diag-suppress 550 \
    -I.

# ======================
# Run the Python benchmark
# ======================
echo "Running Gaussian benchmark..."
python3 gaussian_benchmark.py

echo "Job completed."