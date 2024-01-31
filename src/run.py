from comm.comm import CommManager
from comm.iperf_server import IperfServer
from comm.iperf_client import IperfClient
import utilities.utils as utils
from mab.runner import MabRunner


def run():

    # TODO: handle the params with a config file
    # Runner does not need the netlink comm channel for runtime communication anymore
    # Runner declare and passes the comm channel obj to the environment class
    runner = MabRunner() # pass the last checkpoint filepath to continue training
    runner.setup_communication()
    runner.train()
    runner.stop_communication()
    
    runner.save_history()
    runner.save_model()

    runner.shut_down_env()

if __name__ == '__main__':
    run()