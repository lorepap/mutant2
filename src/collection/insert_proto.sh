#!/bin/bash

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <protocol_name>"
    exit 1
fi

protocol_name=$1

# Add "_mod" to the protocol name
protocol_name="${protocol_name}_mod"

# Get the current script's directory
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Function to find the "src" directory by navigating up the tree
find_src_directory() {
    local dir="$1"
    while [ "$dir" != "/" ]; do
        if [ -d "$dir/src" ]; then
            echo "$dir/src"
            return
        fi
        dir=$(dirname "$dir")
    done
    return 1
}

# Find the "src" directory from the current script's location
src_folder=$(find_src_directory "$script_dir")

if [ -z "$src_folder" ]; then
    echo "Error: 'src' directory not found."
    exit 1
fi

# Change to the "protocol" folder under the "src" directory
protocol_folder="$src_folder/kernel/protocol"
cd "$protocol_folder" || exit 1

# Build and insert kernel module
echo "--- Building and inserting kernel module file ---"
echo ''
sudo make clean
sudo make
sudo insmod $protocol_name.ko

echo ''
# Set the given protocol as congestion control protocol
echo "-- Set $protocol_name as congestion control protocol"
echo ''
sudo sysctl net.ipv4.tcp_congestion_control=$protocol_name || exit 1
