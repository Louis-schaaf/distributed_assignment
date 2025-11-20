#include <stdio.h>
#include <cuda_runtime.h>
#include <math.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

__global__ void gaussianKernel(float* img, float* out, int width, int height) {
    const float kernel[3][3] = {
        {1/16.f, 2/16.f, 1/16.f},
        {2/16.f, 4/16.f, 2/16.f},
        {1/16.f, 2/16.f, 1/16.f}
    };

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    float sum[3] = {0, 0, 0};
    for (int ky = -1; ky <= 1; ky++) {
        for (int kx = -1; kx <= 1; kx++) {
            int ix = min(max(x + kx, 0), width - 1);
            int iy = min(max(y + ky, 0), height - 1);
            int idx = (iy * width + ix) * 3;
            float k = kernel[ky + 1][kx + 1];
            sum[0] += k * img[idx];
            sum[1] += k * img[idx + 1];
            sum[2] += k * img[idx + 2];
        }
    }

    int outIdx = (y * width + x) * 3;
    out[outIdx] = sum[0];
    out[outIdx + 1] = sum[1];
    out[outIdx + 2] = sum[2];
}

int main(int argc, char** argv) {
    if (argc < 3) {
        printf("Usage: %s <input_image> <output_image>\n", argv[0]);
        return 1;
    }

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

    dim3 block(16, 16);
    dim3 grid((width + 15)/16, (height + 15)/16);
    gaussianKernel<<<grid, block>>>(img, out, width, height);
    cudaDeviceSynchronize();

    unsigned char* out_uc = (unsigned char*)malloc(num_pixels * 3);
    for (size_t i = 0; i < num_pixels * 3; i++)
        out_uc[i] = (unsigned char)fminf(fmaxf(out[i], 0), 255);

    stbi_write_png(argv[2], width, height, 3, out_uc, width * 3);

    cudaFree(img);
    cudaFree(out);
    free(out_uc);

    printf("Output saved to %s\n", argv[2]);
    return 0;
}
