# Linux Kernel Patch Installation Guide

## Overview
This guide provides steps to install and apply a patch to the Linux kernel (version 5.4.231) and how to revert it if necessary. The system has been tested on Ubuntu 20.04.

## Prerequisites
- Ubuntu 20.04 or similar Linux distribution
- `git` (for version control)
- `build-essential` and other required packages for kernel compilation
- The Linux kernel source code (5.4.231)

## Steps to Install the Linux Kernel and Apply a Patch

### 1. Install Required Packages
Before starting, install the necessary packages:

```bash
sudo apt update
sudo apt install build-essential libncurses-dev bison flex libssl-dev libelf-dev dwarves
```

### 2. Download the Linux Kernel Source
Download and extract the Linux kernel source:

```bash
cd ~
wget https://cdn.kernel.org/pub/linux/kernel/v5.x/linux-5.4.231.tar.xz
tar xvf linux-5.4.231.tar.xz
cd linux-5.4.231
```

### 3. Apply the Patch
Assuming the patch file is located in the `~/mutant/linux/` directory:

```bash
sudo patch -p1 < ~/mutant/linux/linux-5.4.231-patch.patch
```

### 4. Configure the Kernel
Copy the existing config file and make any necessary changes:

```bash
cp /boot/config-$(uname -r) .config
make menuconfig
```

### 5. Build the Kernel
Compile the kernel (this may take some time):

```bash
make -j$(nproc)
```

### 6. Install Modules and Kernel
Install the modules and the new kernel:

```bash
sudo make modules_install
sudo make install
```

### 7. Update Bootloader
Update GRUB to include the new kernel:

```bash
sudo update-grub
```

### 8. Reboot
Reboot your system to use the new patched kernel:

```bash
sudo reboot
```

