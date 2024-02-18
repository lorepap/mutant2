import subprocess
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--time", type=str, default="30", help="Experiment running time (s)")
parser.add_argument("--proto", type=str, default=None, help="Protocol to use")
args = parser.parse_args()
time = args.time

# Define the experiments based on provided parameters
experiments = [
    {"name": "Baseline", "bw": 42, "rtt": 20, "bdp_mult": 1},
    {"name": "Low_Bandwidth", "bw": 6, "rtt": 20, "bdp_mult": 1},
    {"name": "High_RTT", "bw": 12, "rtt": 80, "bdp_mult": 1},
    {"name": "Large_Queue", "bw": 12, "rtt": 20, "bdp_mult": 10},
    {"name": "Mixed_Conditions", "bw": 42, "rtt": 30, "bdp_mult": 2},
    {"name": "Challenging_Network", "bw": 6, "rtt": 100, "bdp_mult": 1},
    {"name": "Challenging_Network_2", "bw": 12, "rtt": 30, "bdp_mult": 0.5},
]

cmd = ["src/run_collection.py", "--proto", args.proto] if args.proto is not None else ["src/collect.py"]

# Iterate over experiments and run each one
for experiment in experiments:
    command = ["python"] + cmd + \
        [ 
        "--time",
        time, 
        "--bdp_mult",
        str(experiment["bdp_mult"]),
        "--rtt",
        str(experiment["rtt"]),
        "--bw",
        str(experiment["bw"]),
        ]
    
    print(f"\nRunning Experiment: {experiment['name']}")
    subprocess.run(command)

print("\nAll experiments completed.")
