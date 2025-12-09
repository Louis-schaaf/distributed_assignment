#include <iostream>
#include <cmath>
#include <vector>
#include <chrono>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cout << "Usage: " << argv[0] << " <input> <output>\n";
        return 1;
    }

    int width, height, channels;
    unsigned char* img_uc = stbi_load(argv[1], &width, &height, &channels, 3);
    if (!img_uc) {
        std::cout << "Error loading image\n";
        return 1;
    }

    const int R = 10;                 // 21×21 kernel
    const float sigma = 10.f;
    const float sigma2 = 2 * sigma * sigma;

    size_t num_pixels = width * height;
    std::vector<float> img(num_pixels * 3);
    std::vector<float> out(num_pixels * 3);

    // Convert to float
    for (size_t i = 0; i < num_pixels * 3; i++)
        img[i] = img_uc[i];

    stbi_image_free(img_uc);

    // Timing start
    auto t0 = std::chrono::high_resolution_clock::now();

    // === SERIËLE GAUSSIAN BLUR ===
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {

            float sum[3] = {0, 0, 0};
            float total = 0.f;

            for (int ky = -R; ky <= R; ++ky) {
                for (int kx = -R; kx <= R; ++kx) {

                    float w = std::exp(-(kx*kx + ky*ky) / sigma2);

                    int ix = std::min(std::max(x + kx, 0), width - 1);
                    int iy = std::min(std::max(y + ky, 0), height - 1);
                    int idx = (iy * width + ix) * 3;

                    sum[0] += w * img[idx];
                    sum[1] += w * img[idx + 1];
                    sum[2] += w * img[idx + 2];
                    total  += w;
                }
            }

            int outIdx = (y * width + x) * 3;
            out[outIdx]     = sum[0] / total;
            out[outIdx + 1] = sum[1] / total;
            out[outIdx + 2] = sum[2] / total;
        }
    }

    // Timing stop
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::cout << "CPU serial time: " << ms << " ms\n";

    // Convert back to unsigned char
    std::vector<unsigned char> out_uc(num_pixels * 3);
    for (size_t i = 0; i < num_pixels * 3; i++)
        out_uc[i] = (unsigned char) std::min(std::max(out[i], 0.f), 255.f);

    stbi_write_png(argv[2], width, height, 3, out_uc.data(), width * 3);

    return 0;
}