import os

# Get the project root (where this script is located)
project_root = os.path.dirname(os.path.abspath(__file__))

# Path to config.txt
config_path = os.path.join(project_root, "configs", "config.txt")
config = {}
with open(config_path) as f:
    for line in f:
        if '=' in line:
            k, v = line.strip().split('=', 1)
            config[k.strip()] = v.strip()

results_dir = config.get("results_dir", "Results")
datatype = config.get("datatype", "float")
bits = config.get("bits", "12")

# Write CMake fragment
with open(os.path.join(project_root, "config.cmake"), "w") as f:
    f.write(f'set(RESULTS_DIR "{results_dir}")\n')
    f.write(f'set(DATATYPE "{datatype}")\n')
    f.write(f'set(BITS "{bits}")\n')

# Write CMake config header template
with open(os.path.join(project_root, "cmake_config.h.in"), "w") as f:
    f.write("#pragma once\n")
    f.write(f'#define RESULTS_DIR "@RESULTS_DIR@"\n')
    f.write(f'#define DATATYPE "@DATATYPE@"\n')
    f.write(f'#define BITS "@BITS@"\n')
