import subprocess
import datetime

blocks_list = [1, 2, 4, 8, 16, 32, 64, 128]

input_img = "img/in/4k.jpg"

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
logfile = f"logs/cuda_gaussian_bench_{timestamp}.txt"

with open(logfile, "w") as f:

    for b in blocks_list:
        # unoptimized kernel
        cmd_unop = [
            "python", "image_pipeline.py",
            "--filter", "gaussian",
            "--mode", "unop",
            "--blocks", str(b),
            input_img,
            f"img_out/out_unop_{b}.png"
        ]
        proc = subprocess.Popen(cmd_unop, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
        output = proc.communicate()[0]
        f.write(output)

        # extract first line for logs
        # first_line = output.splitlines()[0] if output else ""
        # f.write(first_line + "\n")

        # pre computed kernel
        cmd_pre = [
            "python", "image_pipeline.py",
            "--filter", "gaussian",
            "--mode", "precomp",
            "--blocks", str(b),
            input_img,
            f"img_out/out_pre_{b}.png"
        ]
        proc = subprocess.Popen(cmd_pre, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
        output = proc.communicate()[0]
        f.write(output)

        # extract first line for logs
        # first_line = output.splitlines()[0] if output else ""
        # f.write(first_line + "\n")

print("\nKlaar! Alle resultaten zijn opgeslagen in:")
print(logfile)