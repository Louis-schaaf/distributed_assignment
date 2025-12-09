import subprocess
import datetime

blocks_list = [1, 2, 4, 8, 16, 32, 64, 128]

input_img = "4k.jpg"

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
logfile = f"cuda_gaussian_bench_{timestamp}.txt"

with open(logfile, "w") as f:

    for b in blocks_list:
        f.write(f"\n===== BLOCKS = {b} =====\n")

        # Niet-geoptimaliseerde kernel
        f.write("\n--- gaussian_param_unop ---\n")
        cmd_unop = [
            "python", "image_pipeline.py",
            "--filter", "gaussian",
            "--mode", "unop",
            "--blocks", str(b),
            input_img,
            f"out_unop_{b}.png"
        ]
        proc = subprocess.Popen(cmd_unop, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        output = proc.communicate()[0]
        f.write(output)

        # Geoptimaliseerde kernel
        f.write("\n--- gaussian_param_precomp ---\n")
        cmd_pre = [
            "python", "image_pipeline.py",
            "--filter", "gaussian",
            "--mode", "precomp",
            "--blocks", str(b),
            input_img,
            f"out_pre_{b}.png"
        ]
        proc = subprocess.Popen(cmd_pre, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        output = proc.communicate()[0]
        f.write(output)

print("\nKlaar! Alle resultaten zijn opgeslagen in:")
print(logfile)