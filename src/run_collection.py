from argparse import ArgumentParser
from collection.collector import Collector


def run(args):
    # Collection process
    runner = Collector(protocol=args.proto, 
                    n_steps=args.steps, 
                    log_dir='log/collection', 
                    rtt=args.rtt, 
                    bw=args.bw,
                    bdp_mult=args.bdp_mult, 
                    normalize=args.normalize)
    # Setup connection with the kernel and setup client-server communication (iperf + mahimahi)
    runner.setup_communication()
    # Collect data
    runner.run_collection()
    # Stop communication
    runner.stop_communication()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--proto", type=str, default="cubic", help="Protocol to run for data collection. \
        Supported protocols: cubic, bbr, vegas, veno, westwood, hybla, cdg, illinois, bic, highspeed, htcp, base")
    parser.add_argument("--steps", type=float, default=100, help="Experiment running time (s)")
    parser.add_argument("--rtt", type=float, default=20, help="RTT (ms)")
    parser.add_argument("--bw", type=float, default=12, help="Bandwidth (Mbps)")
    parser.add_argument("--bdp_mult", type=float, default=1, help="BDP multiplier")
    parser.add_argument("--normalize", action='store_true', help="Normalize the reward")
    args = parser.parse_args()
    run(args)