#!/bin/bash
# Define the directory containing the timestamps
timestamps_dir="log/mab/mahimahi"

# Check if the directory exists
if [ ! -d "$timestamps_dir" ]; then
    echo "Error: Directory $timestamps_dir does not exist."
    exit 1
fi

# Iterate over each timestamp folder
for timestamp_folder in "$timestamps_dir"/*; do
    # Check if the item is a directory
    if [ -d "$timestamp_folder" ]; then
        reset;
        # Extract the timestamp from the folder name
        timestamp=$(basename "$timestamp_folder")
        # Call the analyze.sh script with the timestamp as an argument
        ./analyze.sh "$timestamp"
    fi
done
