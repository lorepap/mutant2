import subprocess

# Define the experiments based on provided parameters
experiments = [
    {"name": "Baseline", "bw": 42, "rtt": 20, "bdp_mult": 1},
    {"name": "Low_Bandwidth", "bw": 6, "rtt": 20, "bdp_mult": 1},
    {"name": "High_RTT", "bw": 12, "rtt": 80, "bdp_mult": 1},
    {"name": "Large_Queue", "bw": 12, "rtt": 20, "bdp_mult": 10},
    {"name": "Mixed_Conditions", "bw": 42, "rtt": 30, "bdp_mult": 2},
    {"name": "Challenging_Network", "bw": 6, "rtt": 100, "bdp_mult": 1},
]

# Iterate over experiments and run each one
for experiment in experiments:
    command = [
        "python",
        "src/collect.py",
        "--time",
        "30",
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
