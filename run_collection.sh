#!/bin/bash

source .env/bin/activate

steps=$1
proto=$2
bw=$3
rtt=$4
bdp_mult=$5
bw_factor=$6

# print usage if not enough arguments
if [ $# -ne 6 ]; then
    echo "Usage: $0 <steps> <proto> <bw> <rtt> <bdp_mult> <bw_factor>"
    exit 1
fi

reset;

sudo "$(which python)" src/run_collection.py --steps $steps --proto $proto --bw $bw --rtt $rtt --bdp_mult $bdp_mult --bw_factor $bw_factor
