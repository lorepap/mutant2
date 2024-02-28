import subprocess
from argparse import ArgumentParser
import traceback
import time

import src.collection_plots as plt

parser = ArgumentParser()
parser.add_argument("--time", type=str, default="30", help="Experiment running time (s)")
parser.add_argument("--proto", type=str, nargs='+', default=['bbr'], help="Protocol to use")
parser.add_argument("--experiment", type=str, nargs='+', default=None, help="Experiment to run")
parser.add_argument("--normalize", action='store_true', help="Normalize the reward")
args = parser.parse_args()
collection_time = args.time

# Define the experiments based on provided parameters
experiments = {
    "Baseline": {"bw": 42, "rtt": 20, "bdp_mult": 1},
    "Low_Bandwidth": {"bw": 6, "rtt": 20, "bdp_mult": 1},
    "High_RTT": {"bw": 12, "rtt": 80, "bdp_mult": 1},
    "Large_Queue": {"bw": 12, "rtt": 20, "bdp_mult": 10},
    "Mixed_Conditions": {"bw": 42, "rtt": 30, "bdp_mult": 2},
    "Challenging_Network": {"bw": 6, "rtt": 100, "bdp_mult": 1},
    "Challenging_Network_2": {"bw": 12, "rtt": 30, "bdp_mult": 0.5},
}

if args.experiment is not None:
    test_exp = {exp: experiments[exp] for exp in args.experiment}
else:
    test_exp = experiments


# Iterate over experiments and run each one
for exp_name, exp_settings in test_exp.items():
    for protocol in args.proto:
        # Print protocol information
        print(f"Running {protocol} | Experiment: {exp_name}")

        # Set a maximum number of retries
        max_retries = 20
        retries = 0

        # Loop to retry the subprocess in case of an error
        while retries < max_retries:

            cmd = ["python"] + \
                ["src/run_collection.py", "--proto", protocol] + \
                [ 
                "--time",
                str(collection_time), 
                "--bdp_mult",
                str(exp_settings["bdp_mult"]),
                "--rtt",
                str(exp_settings["rtt"]),
                "--bw",
                str(exp_settings["bw"]),
                "--normalize" if args.normalize else None,
                ]
            # Delete None arguments
            cmd = [arg for arg in cmd if arg is not None]
            
            try:
                subprocess.run(cmd, check=True)
                break  # Break the loop if subprocess runs successfully
            except subprocess.CalledProcessError as e:
                print('\nError in subprocess:')
                print(e)
                retries += 1
                if retries < max_retries:
                    print(f"Retrying... (Attempt {retries}/{max_retries})")
                    time.sleep(1)  # Add a small delay before retrying

            except Exception as e:
                print('\nUnexpected error:')
                print(traceback.format_exc())
                raise RuntimeError(f'Unexpected error for protocol {protocol}')

            plt.plot_thr_single(protocol, exp_settings)
            plt.plot_rtt_single(protocol, exp_settings)
            plt.plot_reward_single(protocol, exp_settings)
        
        if retries == max_retries:
            raise RuntimeError(f'Failed to run subprocess for protocol {protocol} after {max_retries} attempts.')

    if len(args.proto) > 1:
        plt.plot_thr_multi(args.proto, exp_settings)
        plt.plot_rtt_multi(args.proto, exp_settings)
        plt.plot_reward_multi(args.proto, exp_settings)

print("\nAll experiments completed.")