import subprocess
import yaml
from argparse import ArgumentParser
import traceback
import time

# Read protocols from YAML file
with open("config/protocols.yml", "r") as file:
    protocols_data = yaml.safe_load(file)

parser = ArgumentParser()
parser.add_argument("--time", type=float, default=10, help="Experiment running time (s)")
parser.add_argument("--bw", type=float, default=12, help="Bandwidth (Mbps)")
parser.add_argument("--rtt", type=float, default=20, help="RTT (ms)")
parser.add_argument("--bdp_mult", default=100, help="BDP multiplier")
args = parser.parse_args()

# Loop over protocols
for protocol_name, protocol_info in protocols_data.items():
    # Print protocol information
    print(f"Running {protocol_name}..")

    # Set a maximum number of retries
    max_retries = 20
    retries = 0

    # Loop to retry the subprocess in case of an error
    while retries < max_retries:
        # Run the python script with specified arguments
        cmd = [
            'python',
            'src/run_collection.py',
            '--proto',
            protocol_name,
            '--time',
            str(args.time),
            '--bw',
            str(args.bw),
            '--rtt',
            str(args.rtt),
            '--bdp_mult',
            str(args.bdp_mult)
        ]

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
            raise RuntimeError(f'Unexpected error for protocol {protocol_name}')

    if retries == max_retries:
        raise RuntimeError(f'Failed to run subprocess for protocol {protocol_name} after {max_retries} attempts.')
