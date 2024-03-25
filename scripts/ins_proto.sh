#!/bin/bash

cd ~/linux-5.4.231-mutant/net/ipv4
echo '--- Inserting protocols as modules ---'
sudo insmod tcp_hybla.ko
sudo insmod tcp_bbr.ko
sudo insmod tcp_westwood.ko
sudo insmod tcp_veno.ko
sudo insmod tcp_vegas.ko
sudo insmod tcp_yeah.ko
sudo insmod tcp_cdg.ko
sudo insmod tcp_bic.ko
sudo insmod tcp_htcp.ko
sudo insmod tcp_highspeed.ko
sudo insmod tcp_illinois.ko
