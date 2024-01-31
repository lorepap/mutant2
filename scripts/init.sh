#!/bin/bash

# Run init_kernel.sh
echo '--- Running init_kernel.sh'
sh scripts/init_kernel.sh || exit 1
echo ''

echo '--- Running mahimahi.sh'
sh scripts/mahimahi.sh || exit 1
echo ''