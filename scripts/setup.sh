#!/bin/bash

# install needed modules
echo '--- Installing needed system modules ---'
echo ''
cd src && sudo apt-get install $(cat modules.txt) || exit 1

echo '-- Updating locate database'
echo ''
sudo updatedb

echo ''

# install header file
# echo '--- Installing header files for your linux kernel ---'
# echo ''
# sudo apt-get install linux-headers-$(uname -r) || exit 1

echo ''

# create virtual environment
echo '--- Creating virtual python environment ---'
echo ''
python3 -m venv venv || exit 1

echo ''

# enable virtual environment
echo '--- change directory to src & run source venv/bin/activate ---'
exit 0



