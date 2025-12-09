#include <stdio.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdlib.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb/stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb/stb_image_write.h"

__global__ void gaussianKernel(float* img, float* out, int width, int height) {
    const int R = 10;
    const float sigma = 10.f;
    const float sigma2 = 2 * sigma * sigma;

    int local_tid = threadIdx.y * blockDim.x + threadIdx.x;
    int tid = blockIdx.x * (blockDim.x * blockDim.y) + local_tid;
    int total_threads = gridDim.x * blockDim.x * blockDim.y;

    int num_pixels = width * height;

    for (int p = tid; p < num_pixels; p += total_threads) {
        int x = p % width;
        int y = p / width;

        float sum[3] = {0,0,0};
        float total = 0.f;

        for (int ky = -R; ky <= R; ++ky) {
            for (int kx = -R; kx <= R; ++kx) {
                float w = expf(-(kx*kx + ky*ky) / sigma2);

                int ix = min(max(x + kx, 0), width - 1);
                int iy = min(max(y + ky, 0), height - 1);
                int idx = (iy * width + ix) * 3;

                sum[0] += w * img[idx];
                sum[1] += w * img[idx + 1];
                sum[2] += w * img[idx + 2];
                total  += w;
            }
        }

        int outIdx = p * 3;
        out[outIdx]     = sum[0] / total;
        out[outIdx + 1] = sum[1] / total;
        out[outIdx + 2] = sum[2] / total;
    }
}

int main(int argc, char** argv) {
    if (argc < 4) {
        printf("Usage: %s <input_image> <output_image> <num_blocks>\n", argv[0]);
        printf("Example: %s input.png output.png 4\n", argv[0]);
        return 1;
    }

    // Parse number of blocks from command line
    int num_blocks = atoi(argv[3]);
    if (num_blocks <= 0) {
        printf("Error: Number of blocks must be positive\n");
        return 1;
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int width, height, channels;
    unsigned char* img_uc = stbi_load(argv[1], &width, &height, &channels, 3);
    if (!img_uc) return 1;

    size_t num_pixels = width * height;
    size_t size = num_pixels * 3 * sizeof(float);

    float *img, *out;
    cudaMallocManaged(&img, size);
    cudaMallocManaged(&out, size);

    for (size_t i = 0; i < num_pixels * 3; i++)
        img[i] = (float)img_uc[i];

    free(img_uc);

    dim3 block(32, 32);
    dim3 grid(num_blocks, 1);

    // Timing
    cudaEventRecord(start);
    gaussianKernel<<<grid, block>>>(img, out, width, height);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    unsigned char* out_uc = (unsigned char*)malloc(num_pixels * 3);
    // Convert float output back to unsigned char
    for (size_t i = 0; i < num_pixels * 3; i++)
        out_uc[i] = (unsigned char)fminf(fmaxf(out[i], 0), 255);

    // Save output image
    stbi_write_png(argv[2], width, height, 3, out_uc, width * 3);

    // Free GPU memory
    cudaFree(img);
    cudaFree(out);
    free(out_uc);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    printf("unoptimized,%s,%d,%f\n", argv[2], num_blocks, ms);
    printf("Kernel time: %f ms\n", ms);
    printf("Blocks: %d, Threads per block: %d\n", num_blocks, block.x * block.y);
    printf("Output saved to %s\n", argv[2]);
    return 0;
}