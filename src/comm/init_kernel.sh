#!/bin/bash

base_path=$(pwd)/src
cd $base_path/kernel
echo ''

# Build and insert kernel module
echo '--- Building and inserting kernel module file ---'
echo ''
sudo make clean
sudo make
# sudo /usr/src/linux-$(uname -r)/scripts/sign-file sha256 ./key/MOK.priv ./key/MOK.der mimic.ko
sudo insmod mutant.ko

echo ''
# Set mutant as congestion control protocol
echo '-- Set mutant as congestion control protocol'
echo ''
sudo sysctl net.ipv4.tcp_congestion_control=mutant || exit 1