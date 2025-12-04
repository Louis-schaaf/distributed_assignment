import numpy as np
import subprocess
import argparse
import imageio
import os

def save_kernel_txt(kernel, path):
    k = kernel.shape[0]
    with open(path, "w") as f:
        f.write(f"{k}\n")
        for row in kernel:
            f.write(" ".join(str(v) for v in row) + "\n")

def run_cuda(kernel_path, input_path, output_path, blocks):
    subprocess.run([
        "./gaussian_param",
        input_path,
        output_path,
        str(blocks),
        kernel_path
    ], check=True)

def gaussian_kernel(size, sigma):
    ax = np.linspace(-(size//2), size//2, size)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2*sigma*sigma))
    kernel /= kernel.sum()
    return kernel

def sobel_x():
    return np.array([
        [-1,0,1],
        [-2,0,2],
        [-1,0,1]
    ], dtype=float)

def sobel_y():
    return np.array([
        [-1,-2,-1],
        [0,  0,  0],
        [1,  2,  1]
    ], dtype=float)

def pipeline_gaussian(input_img, output_img, args, sigma=1.0):
    if args.kernel is not None:
        run_cuda(args.kernel, input_img, output_img, args.blocks)
        return
    K = gaussian_kernel(5, sigma)
    save_kernel_txt(K, "kernel.txt")
    run_cuda("kernel.txt", input_img, output_img, args.blocks)

def pipeline_sobel(input_img, output_img, args):
    if args.kernel is not None:
        run_cuda(args.kernel, input_img, output_img, args.blocks)
        return
    Kx = sobel_x()
    save_kernel_txt(Kx, "kernel.txt")
    run_cuda("kernel.txt", input_img, "sobel_x.png", args.blocks)

    Ky = sobel_y()
    save_kernel_txt(Ky, "kernel.txt")
    run_cuda("kernel.txt", input_img, "sobel_y.png", args.blocks)

def pipeline_canny(input_img, output_img, args):
    if args.kernel is not None:
        run_cuda(args.kernel, input_img, output_img, args.blocks)
        return
    # Gaussian blur
    K = gaussian_kernel(5, 1.4)
    save_kernel_txt(K, "kernel.txt")
    run_cuda("kernel.txt", input_img, "blur.png", args.blocks)

    # Sobel X
    Kx = sobel_x()
    save_kernel_txt(Kx, "kernel.txt")
    run_cuda("kernel.txt", "blur.png", "grad_x.png", args.blocks)

    # Sobel Y
    Ky = sobel_y()
    save_kernel_txt(Ky, "kernel.txt")
    run_cuda("kernel.txt", "blur.png", "grad_y.png", args.blocks)

    gx = imageio.imread("grad_x.png").astype(float)
    gy = imageio.imread("grad_y.png").astype(float)

    # Gradient magnitude and angle
    mag = np.sqrt(gx*gx + gy*gy)
    angle = np.arctan2(gy, gx)

    # Normalize angle to 0–180 degrees
    angle = np.rad2deg(angle)
    angle[angle < 0] += 180

    # Non-maximum suppression
    nms = np.zeros_like(mag)
    h, w = mag.shape[:2]

    for y in range(1, h-1):
        for x in range(1, w-1):
            a = angle[y, x]  # scalar

            if (0 <= a < 22.5) or (157.5 <= a <= 180):
                n1, n2 = mag[y, x-1], mag[y, x+1]     # horizontal
            elif 22.5 <= a < 67.5:
                n1, n2 = mag[y-1, x+1], mag[y+1, x-1] # diag 1
            elif 67.5 <= a < 112.5:
                n1, n2 = mag[y-1, x], mag[y+1, x]     # vertical
            else:
                n1, n2 = mag[y-1, x-1], mag[y+1, x+1] # diag 2

            if mag[y, x] >= n1 and mag[y, x] >= n2:
                nms[y, x] = mag[y, x]

    # Double threshold
    high = nms.max() * 0.2
    low = high * 0.5

    strong = (nms >= high)
    weak = (nms >= low) & ~strong

    result = np.zeros_like(nms, dtype=np.uint8)
    result[strong] = 255
    result[weak] = 75

    # Hysteresis
    for y in range(1, h-1):
        for x in range(1, w-1):
            if result[y, x] == 75:
                if np.any(result[y-1:y+2, x-1:x+2] == 255):
                    result[y, x] = 255
                else:
                    result[y, x] = 0

    imageio.imwrite(output_img, result)
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--filter", type=str, required=True)
    parser.add_argument("input")
    parser.add_argument("output")
    parser.add_argument("--blocks", type=int, default=1024, help="Number of CUDA blocks")
    parser.add_argument("--kernel", type=str, default=None, help="Optional path to custom kernel .txt file")
    args = parser.parse_args()

    custom_kernel = args.kernel
    # If a custom kernel is provided
    if custom_kernel is not None:
        run_cuda(custom_kernel, args.input, args.output, args.blocks)
        exit(0)

    if args.filter == "gaussian":
        pipeline_gaussian(args.input, args.output, args)
    elif args.filter == "sobel":
        pipeline_sobel(args.input, args.output, args)
    elif args.filter == "canny":
        pipeline_canny(args.input, args.output, args)
    else:
        print("Unknown filter")