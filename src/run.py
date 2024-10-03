from argparse import ArgumentParser 
from utilities.selection_strategy import MPTSStrategy, ManualSelectionStrategy
import os 
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from mab.runner import MabRunner

def run(args):
    # TODO: handle the params with a config file
    # Runner does not need the netlink comm channel for runtime communication anymore
    # Runner declare and passes the comm channel obj to the environment class
    
    selection = MPTSStrategy if args.mpts else ManualSelectionStrategy
    runner = MabRunner(min_rtt=args.rtt, bw=args.bw, bdp_mult=int(args.bdp_mult), bw_factor=int(args.bw_factor), 
            restore=args.restore, protocol_selection_strategy=selection)
    # runner.setup_communication()
    runner.train()
    # runner.stop_communication()
    print("Training done")

if __name__ == '__main__':
    parser = ArgumentParser()
    # Accept a list of policies to be used in the environment - if it's not passed, use all of them
    parser.add_argument('--proto', '-p', nargs='+', default=None, type=str)
    parser.add_argument("--bw", '-bw', type=int, default=12, help="Bandwidth (Mbps)")
    parser.add_argument("--rtt", '-rtt', type=int, default=20, help="RTT (ms)")
    parser.add_argument("--bdp_mult", '-bdp_mult', type=float, default=1, help="BDP multiplier")
    parser.add_argument("--bw_factor", '-bw_factor', type=float, default=1, help="Bandwidth factor")
    parser.add_argument("--mpts", '-mpts', action='store_true', help="Use MPTS")
    parser.add_argument("--k", '-k', type=int, default=4, help="Number of arms")
    parser.add_argument("--restore", '-r', action='store_true', help="Restore last checkpoint")
    parser.add_argument("--test", '-t', action='store_true', help="Run test")
    args = parser.parse_args()
    run(args)