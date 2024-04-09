from argparse import ArgumentParser 
from mab.test_runner import TestRunner


def run(args):
    # TODO: handle the params with a config file
    # Runner does not need the netlink comm channel for runtime communication anymore
    # Runner declare and passes the comm channel obj to the environment class
    runner = TestRunner(int(args.rtt), int(args.bw), int(args.bdp_mult), int(args.bw_factor), int(args.k))
    runner.setup_communication()
    runner.test()
    runner.stop_communication()

    runner.shut_down_env()

if __name__ == '__main__':
    parser = ArgumentParser()
    # Accept a list of policies to be used in the environment - if it's not passed, use all of them
    parser.add_argument("--bw", '-bw', type=int, default=12, help="Bandwidth (Mbps)")
    parser.add_argument("--rtt", '-rtt', type=int, default=20, help="RTT (ms)")
    parser.add_argument("--bdp_mult", '-bdp_mult', type=float, default=1, help="BDP multiplier")
    parser.add_argument("--bw_factor", '-bw_factor', type=float, default=1, help="Bandwidth factor")
    parser.add_argument("--k", '-k', type=int, default=4, help="Number of arms")
    args = parser.parse_args()
    run(args)