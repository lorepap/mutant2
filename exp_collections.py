import subprocess
from argparse import ArgumentParser
import traceback
import time
import src.utilities.utils as utils
from src.utilities.experiments import Experiment
import src.collection_plots as plt

parser = ArgumentParser()
parser.add_argument("--steps", default=100, help="Experiment running time (s)")
parser.add_argument("--proto", type=str, nargs='+', default=None, help="Protocol to use")
parser.add_argument("--experiment", type=str, nargs='+', default=None, help="Experiment to run")
parser.add_argument("--normalize", action='store_true', help="Normalize the reward")
args = parser.parse_args()

if args.experiment is not None:
    test_exp = {exp: Experiment(exp) for exp in args.experiment}
else:
    test_exp = Experiment.experiments()

if not args.proto:
    protos = utils.parse_protocols_config().keys()
else:
    protos = args.proto

print(protos)
# Iterate over experiments and run each one
for exp_name, exp_settings in test_exp.items():
    for protocol in protos:
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
                "--steps",
                str(args.steps), 
                "--bdp_mult",
                str(exp_settings.bdp_mult),
                "--rtt",
                str(exp_settings.rtt),
                "--bw",
                str(exp_settings.bw),
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

    if len(protos) > 1:
        plt.plot_thr_multi(protos, exp_settings, args.steps)
        plt.plot_rtt_multi(protos, exp_settings, args.steps)
        plt.plot_reward_multi(protos, exp_settings, args.steps)

print("\nAll experiments completed.")