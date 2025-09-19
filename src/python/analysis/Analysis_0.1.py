import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load and parse data
with open("output.txt", "r") as file:
    content = file.read()

# Convert comma-separated string to list of floats
float_values = [float(val) for val in content.strip().split(",") if val and float(val) != 0.0]


# Create DataFrame
df = pd.DataFrame(float_values, columns=["Float32"])

#  Print statistics
print("Basic Statistics:")
print(f"Count           : {df['Float32'].count()}")
print(f"Mean            : {df['Float32'].mean():.8f}")
print(f"Median          : {df['Float32'].median():.8f}")
print(f"Standard Dev.   : {df['Float32'].std():.8f}")
print(f"Min             : {df['Float32'].min():.8f}")
print(f"Max             : {df['Float32'].max():.8f}")
print(f"Range           : {(df['Float32'].max() - df['Float32'].min()):.8f}")
print(f"25th Percentile : {df['Float32'].quantile(0.25):.8f}")
print(f"75th Percentile : {df['Float32'].quantile(0.75):.8f}")
print(f"IQR             : {(df['Float32'].quantile(0.75) - df['Float32'].quantile(0.25)):.8f}")
print(f"Skewness        : {df['Float32'].skew():.8f}")
print(f"Kurtosis        : {df['Float32'].kurt():.8f}")

#  Plot histogram with standard deviation lines
plt.figure(figsize=(10, 6))
sns.histplot(df["Float32"], bins=30, kde=True, color='steelblue', edgecolor='black')

mean = df["Float32"].mean()
std = df["Float32"].std()

plt.axvline(mean, color='red', linestyle='--', label=f'Mean: {mean:.4f}')
plt.axvline(mean + std, color='green', linestyle='--', label=f'+1 Std Dev: {(mean + std):.4f}')
plt.axvline(mean - std, color='green', linestyle='--', label=f'-1 Std Dev: {(mean - std):.4f}')

plt.title("Distribution of 32-bit Float Values")
plt.xlabel("Float Value")
plt.ylabel("Frequency")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

