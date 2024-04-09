#!/bin/bash
# Define the directory containing the timestamps
collection_dir="log/collection/mahimahi"

# Check if the directory exists
if [ ! -d "$collection_dir" ]; then
    echo "Error: Directory $timestamps_dir does not exist."
    exit 1
fi

# Iterate over each timestamp folder
for proto_folder in "$collection_dir"/*; do
    echo "Doing Analysis of $proto_folder ..."
    for proto_file in "$proto_folder"/*; do
        # Check if the file starts with "uplink"
        if [[ "$proto_file" == *"uplink"* ]]; then
            echo "Doing Analysis of $proto_file ..."
            # Call the analyze.sh script with the timestamp as an argument
            ./analyze.sh "$proto_file"
        fi
    done
done
