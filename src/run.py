from argparse import ArgumentParser 
from mab.runner import MabRunner


def run(args):

    # TODO: handle the params with a config file
    # Runner does not need the netlink comm channel for runtime communication anymore
    # Runner declare and passes the comm channel obj to the environment class
    runner = MabRunner(args.p, args.rtt, args.bw, args.bdp_mult) # pass the last checkpoint filepath to continue training
    runner.setup_communication()
    runner.train()
    runner.stop_communication()

    runner.save_model()

    runner.shut_down_env()

if __name__ == '__main__':
    parser = ArgumentParser()
    # Accept a list of policies to be used in the environment - if it's not passed, use all of them
    parser.add_argument('-p', nargs='+', default=None, type=str)
    parser.add_argument("-bw", type=int, default=12, help="Bandwidth (Mbps)")
    parser.add_argument("-rtt", type=int, default=20, help="RTT (ms)")
    parser.add_argument("-bdp_mult", type=float, default=1, help="BDP multiplier")
    args = parser.parse_args()
    run(args)