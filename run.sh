#!/bin/bash

# # Array of bandwidth values
bw_values=(12 24 48 96 192)

# Iterate through the values and execute the command
# for bw in "${bw_values[@]}"; do
#   python src/run.py --bw "$bw"
# done

# Values of K
k_values=(8 9)

# Iterate through the values and execute the command
for k in "${k_values[@]}"; do
  python src/run.py -k "$k"
done

