import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

files = [
    "convBigRELU__DiffAI.onnx_cifar_all_gpus.csv",
    "convBigRELU__DiffAI.onnx_cifar_cpu_only.csv",
    "convBigRELU__DiffAI.onnx_cifar_per_operation.csv"
]

labels = ["Direct GPU", "CPU Only", "Per Operation GPU"]

plt.figure(figsize=(10, 6))

for i, file in enumerate(files):
    df = pd.read_csv(file)

    df = df[df["Operation"] != "Total Time"]

    operations = df["Operation"].values
    op_times = df["Op Time (s)"].values

    x = np.arange(len(operations))

    # Small horizontal offset 
    offset = (i - 1) * 0.15
    plt.scatter(x + offset, op_times, s=80, label=labels[i])


plt.xticks(np.arange(len(operations)), operations, rotation=30, ha="right")
plt.xlabel("Operation", fontsize=12)
plt.ylabel("Time (s)", fontsize=12)
plt.title("Operation Time Comparison", fontsize=14, fontweight="bold")

plt.grid(alpha=0.3)
plt.legend(frameon=False)
plt.tight_layout()
plt.savefig("operation_time_comparison.png", dpi=300)
