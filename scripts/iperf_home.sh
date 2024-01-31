#!/bin/bash

dt=$(date '+%Y.%m.%d.%H.%M.%S')
echo ""

# Read in some arguments
read -p 'Time in seconds [60]: ' tt
read -p 'IP Address [192.168.1.50]: ' ip
read -p 'Mahimahi Trace [att.lte.driving]: ' tr

tt=${tt:-60}
ip=${ip:-192.168.1.50}
tr=${tr:-att.lte.driving}

echo "Running iperf3 client with log at: /home/pokorie/Documents/repos/mimic/iperf/$tr.$dt.json"
echo ""

# Run iperf
iperf3 -c $ip  -t $tt -J  --logfile /home/pokorie/Documents/repos/mimic/iperf/$tr.$dt.json
