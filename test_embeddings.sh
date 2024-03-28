#!/bin/bash

# Activate your virtual environment (replace '.env' with the actual name)
source .env/bin/activate 

# Specific steps values to use
steps_values=(10 50 100 500)
bw_values=(24 48 96)
bdp_mult_values=(1 2 6)
bw_factor_values=(1 2 4 8)

# Loop through the values
for bw in "${bw_values[@]}"; do
    python test_embeddings.py --proto bbr --bw $bw --bw_factor 2 --train_steps 300 --embeddings > test_embeddings/log/out/output_bw$bw.txt 2>&1
done

for bdp in "${bdp_mult_values[@]}"; do
    python test_embeddings.py --proto bbr --bw 48 --bw_factor 2 --bdp_mult $bdp --train_steps 300 --embeddings > test_embeddings/log/out/output_bdp$bdp.txt 2>&1
done

for bw_factor in "${bw_factor_values[@]}"; do
    python test_embeddings.py --proto bbr --bw 12 --bw_factor $bw_factor --train_steps 300 --embeddings > test_embeddings/log/out/output_bw_factor$bw_factor.txt 2>&1
done

# for steps in "${steps_values[@]}"; do
#     python test_embeddings.py --proto bbr --bw 12 --bw_factor 2 --train_steps $steps > test_embeddings/log/out/output_$steps.txt 2>&1
# done

# Deactivate the virtual environment (optional)
deactivate 
