#!/bin/bash

# Get the kernel source directory
KERNEL_SOURCE_DIR="/home/lorenzo/Desktop/linux-5.4.231"

# Navigate to the kernel source directory
cd "$KERNEL_SOURCE_DIR" || exit

# Build the kernel
sudo make -j$(nproc)

# Install modules
sudo make modules_install

# Install the kernel
sudo make install

echo "The kernel has been built and installed successfully! Rebooting..."

sudo reboot
