#!/bin/bash

# Activate your virtual environment (replace '.env' with the actual name)
source .env/bin/activate 

# Specific steps values to use
steps_values=(10 50 100 500)

# Loop through the values
for steps in "${steps_values[@]}"; do
    python test_embeddings.py --proto bbr --bw 12 --bw_factor 2 --train_steps $steps > test_embeddings/log/out/output_$steps.txt 2>&1
done

# Deactivate the virtual environment (optional)
deactivate 
