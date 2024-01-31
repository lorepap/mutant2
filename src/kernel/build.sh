sudo make clean
sudo make
sudo /usr/src/linux-$(uname -r)/scripts/sign-file sha256 ./key/MOK.priv ./key/MOK.der mimic.ko
sudo insmod mimic.ko