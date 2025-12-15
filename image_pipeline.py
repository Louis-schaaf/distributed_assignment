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

def run_cuda(kernel_path, input_path, output_path, blocks, mode):
    if mode == "unop":
        # unoptimized version: no kernel file
        subprocess.run([
            "compiled/gaussian_param_unop",
            input_path,
            output_path,
            str(blocks)
        ], check=True)
    else:
        # precomputed version
        subprocess.run([
            "compiled/gaussian_param_precomp",
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

def pipeline_gaussian(input_img, output_img, args):
    if args.mode == "unop":
        # No kernel, direct call
        run_cuda(None, input_img, output_img, args.blocks, args.mode)
        return
    if args.kernel is not None:
        run_cuda(args.kernel, input_img, output_img, args.blocks, args.mode)
        return
    K = gaussian_kernel(21, args.sigma)
    save_kernel_txt(K, "kernels/kernel.txt")
    run_cuda("kernels/kernel.txt", input_img, output_img, args.blocks, args.mode)

def pipeline_sobel(input_img, output_img, args):
    if args.kernel is not None:
        run_cuda(args.kernel, input_img, output_img, args.blocks, args.mode)
        return
    Kx = sobel_x()
    save_kernel_txt(Kx, "kernels/kernel.txt")
    run_cuda("kernels/kernel.txt", input_img, "img/temp/sobel_x.png", args.blocks, args.mode)

    Ky = sobel_y()
    save_kernel_txt(Ky, "kernels/kernel.txt")
    run_cuda("kernels/kernel.txt", input_img, "img/temp/sobel_y.png", args.blocks, args.mode)
def pipeline_canny(input_img, output_img, args):
    if args.kernel is not None:
        run_cuda(args.kernel, input_img, output_img, args.blocks, args.mode)
        return
    # Gaussian blur
    K = gaussian_kernel(5, 1.4)
    save_kernel_txt(K, "kernels/kernel.txt")
    run_cuda("kernels/kernel.txt", input_img, "img/temp/imgblur.png", args.blocks, args.mode)

    # Sobel X
    Kx = sobel_x()
    save_kernel_txt(Kx, "kernels/kernel.txt")
    run_cuda("kernels/kernel.txt", "img/temp/imgblur.png", "img/temp/grad_x.png", args.blocks, args.mode)

    # Sobel Y
    Ky = sobel_y()
    save_kernel_txt(Ky, "kernels/kernel.txt")
    run_cuda("kernels/kernel.txt", "img/temp/imgblur.png", "img/temp/grad_y.png", args.blocks, args.mode)

    gx = imageio.imread("img/temp/grad_x.png").astype(float)
    if gx.ndim == 3:
        gx = gx[:, :, 0]

    gy = imageio.imread("img/temp/grad_y.png").astype(float)
    if gy.ndim == 3:
        gy = gy[:, :, 0]

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
            a = float(angle[y, x])  # force scalar

            if (a >= 0 and a < 22.5) or (a >= 157.5 and a <= 180):
                n1, n2 = mag[y, x-1], mag[y, x+1]
            elif (a >= 22.5 and a < 67.5):
                n1, n2 = mag[y-1, x+1], mag[y+1, x-1]
            elif (a >= 67.5 and a < 112.5):
                n1, n2 = mag[y-1, x], mag[y+1, x]
            else:
                n1, n2 = mag[y-1, x-1], mag[y+1, x+1]

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
    parser.add_argument("--sigma", type=float, default=10.0, help="Sigma value for Gaussian blur")
    parser.add_argument("--mode", type=str, default="precomp", help="Choose gaussian_param_precomp or gaussian_param_unop")
    args = parser.parse_args()

    custom_kernel = args.kernel
    # If a custom kernel is provided
    if custom_kernel is not None:
        run_cuda(custom_kernel, args.input, args.output, args.blocks, args.mode)
        exit(0)

    if args.filter == "gaussian":
        pipeline_gaussian(args.input, args.output, args)
    elif args.filter == "sobel":
        pipeline_sobel(args.input, args.output, args)
    elif args.filter == "canny":
        pipeline_canny(args.input, args.output, args)
    else:
        print("Unknown filter")