import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- Load configuration ---
config = {}
with open("config.txt", "r") as cfg:
    for line in cfg:
        if '=' in line:
            key, value = line.strip().split('=')
            config[key.strip()] = value.strip()

# --- Extract config values ---
try:
    datatype = config["datatype"].strip().lower()
except KeyError:
    raise ValueError("Missing 'datatype' entry in config.txt. Please specify datatype=float, int, uint8, etc.")

try:
    bits = config["bits"].strip()
except KeyError:
    raise ValueError("Missing 'bits' entry in config.txt. Please specify bits=8, bits=16, etc.")

try:
    raw_val = config["exclude_zeros"].strip().lower()
    if raw_val not in ["true", "false"]:
        raise ValueError("Invalid value for 'exclude_zeros'. Must be 'True' or 'False'.")
    exclude_zeros = raw_val == "true"
except KeyError:
    raise ValueError("Missing 'exclude_zeros' entry in config.txt. Please specify exclude_zeros=True or False.")

# --- Extract results_dir ---
try:
    results_dir = config["results_dir"].strip()
except KeyError:
    raise ValueError("Missing 'results_dir' entry in config.txt. Please specify results_dir=YourFolderName")

# --- Create results directory if it doesn't exist ---
if not os.path.exists(results_dir):
    os.makedirs(results_dir)
    print(f"Created results directory: {results_dir}")
else:
    print(f"Using existing results directory: {results_dir}")
    
# --- Construct input filename ---
input_filename = f"output_{datatype}_{bits}_bit.txt"

# --- Load and parse data ---
with open(input_filename, "r") as file:
    content = file.read()

# --- Convert to numeric list ---
raw_values = content.strip().split(",")
try:
    numeric_values = [float(val) for val in raw_values if val.strip()]
except ValueError:
    raise ValueError(f"Failed to convert values to float. Check formatting in {input_filename}")

if exclude_zeros:
    numeric_values = [val for val in numeric_values if val != 0.0]



# --- Create full output path ---
output_dir = os.path.join(results_dir, f"Analysis_{datatype}_{bits}")
os.makedirs(output_dir, exist_ok=True)  # Creates if not exists, does NOT overwrite


# --- Create DataFrame ---
column_name = f"{datatype}{bits}"
df = pd.DataFrame(numeric_values, columns=[column_name])

# --- Save statistics to text file ---
#stats_filename = f"Stats_{datatype}_{bits}_bits.txt"
stats_filename = os.path.join(output_dir, f"Stats_{datatype}_{bits}_bits.txt")
with open(stats_filename, "w") as stats_file:
    stats_file.write("Basic Statistics:\n")
    stats_file.write(f"Count           : {df[column_name].count()}\n")
    stats_file.write(f"Mean            : {df[column_name].mean():.8f}\n")
    stats_file.write(f"Median          : {df[column_name].median():.8f}\n")
    stats_file.write(f"Standard Dev.   : {df[column_name].std():.8f}\n")
    stats_file.write(f"Min             : {df[column_name].min():.8f}\n")
    stats_file.write(f"Max             : {df[column_name].max():.8f}\n")
    stats_file.write(f"Range           : {(df[column_name].max() - df[column_name].min()):.8f}\n")
    stats_file.write(f"25th Percentile : {df[column_name].quantile(0.25):.8f}\n")
    stats_file.write(f"75th Percentile : {df[column_name].quantile(0.75):.8f}\n")
    stats_file.write(f"IQR             : {(df[column_name].quantile(0.75) - df[column_name].quantile(0.25)):.8f}\n")
    stats_file.write(f"Skewness        : {df[column_name].skew():.8f}\n")
    stats_file.write(f"Kurtosis        : {df[column_name].kurt():.8f}\n")

print(f"Statistics saved to: {stats_filename}")


# --- Plot histogram ---
plt.figure(figsize=(10, 6))
sns.histplot(df[column_name], bins=30, kde=True, color='steelblue', edgecolor='black')

mean = df[column_name].mean()
std = df[column_name].std()

plt.axvline(mean, color='red', linestyle='--', label=f'Mean: {mean:.4f}')
plt.axvline(mean + std, color='green', linestyle='--', label=f'+1 Std Dev: {(mean + std):.4f}')
plt.axvline(mean - std, color='green', linestyle='--', label=f'-1 Std Dev: {(mean - std):.4f}')

plt.title(f"Distribution of {datatype}-{bits}-bit Values")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.legend()
plt.grid(True)
plt.tight_layout()

# --- Save plot ---
#output_plot = f"Result_{datatype}_{bits}_bits.png"
output_plot = os.path.join(output_dir, f"Result_{datatype}_{bits}_bits.png")
stats_filename = os.path.join(output_dir, f"Stats_{datatype}_{bits}_bits.txt")

plt.savefig(output_plot)
print(f"\nPlot saved as: {output_plot}")

plt.show()

