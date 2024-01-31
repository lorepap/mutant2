#!/bin/bash

sudo sysctl -w net.ipv4.ip_forward=1 
# mm-link /usr/share/mahimahi/traces/ATT-LTE-driving.up /usr/share/mahimahi/traces/ATT-LTE-driving.down 
# mm-link /usr/share/mahimahi/traces/ATT-LTE-driving-2016.up /usr/share/mahimahi/traces/ATT-LTE-driving-2016.down 

# mm-link /usr/share/mahimahi/traces/TMobile-LTE-short.up /usr/share/mahimahi/traces/TMobile-LTE-short.down 
# mm-link /usr/share/mahimahi/traces/TMobile-LTE-driving.up /usr/share/mahimahi/traces/TMobile-LTE-driving.down 
# mm-link /usr/share/mahimahi/traces/TMobile-UMTS-driving.up /usr/share/mahimahi/traces/TMobile-UMTS-driving.down 

mm-link /usr/share/mahimahi/traces/Verizon-LTE-short.up /usr/share/mahimahi/traces/Verizon-LTE-short.down 
# mm-link /usr/share/mahimahi/traces/Verizon-LTE-driving.up /usr/share/mahimahi/traces/Verizon-LTE-driving.down 
# mm-link /usr/share/mahimahi/traces/Verizon-EVDO-driving.up /usr/share/mahimahi/traces/Verizon-EVDO-driving.down 