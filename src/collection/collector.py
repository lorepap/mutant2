import os
import yaml
import time

from collection.kernel_comm import CollectionCommManager
from comm.kernel_thread import KernelRequest


# Protocol mapping
PROTOCOL_MAPPING = {
    "cubic": 0,
    "hybla": 1,
    "bbr": 2,
    "westwood": 3,
    "veno": 4,
    "vegas": 5,
    "yeah": 6,
    "cdg": 7,
    "bic": 8,
    "htcp": 9,
    "highspeed": 10,
    "illinois": 11,
}


class Collector():
    """ Collector class
    The collector runs a data collection campaign by running a specific protocol for a predefined time period.
    It setup a communication with Mutant kernel module (client) to collect the traffic data (network statistics).
    The data collected are stored locally as a csv file.

    Inputs: protocol, data collection time (running_time).
    Output: csv file of data collected
    """

    def __init__(self, protocol, running_time, log_dir='log/collection', rtt=20, bw=12, bdp_mult=1):
        self.bw = int(bw)
        self.rtt = int(rtt)
        self.bdp_mult = round(bdp_mult, 1) if bdp_mult < 1 else int(bdp_mult)
        
        self.cm = CollectionCommManager(log_dir_name=log_dir, client_time=running_time, rtt=rtt,
                                bw=bw, bdp_mult=bdp_mult)
        self.log_dir = log_dir
        self.protocol = protocol
        self.proto_id = PROTOCOL_MAPPING.get(protocol.lower())  # Convert to lowercase for case-insensitivity
        self.running_time = running_time
        # TODO: handle the params with a config file
        with open('config/train.yml', 'r') as file:
            config = yaml.safe_load(file)

        self.num_fields_kernel = config['num_fields_kernel']
        self.initiated = False
        self._init_communication()

    def setup_communication(self):
        # Set up iperf client-server communication
        # Now a single flow between client and server is running
        # We can now set up the runner and start training the RL model    
        self.cm.init_kernel_communication()
        self.cm.start_communication(client_tag=f'{self.protocol}', server_log_dir=os.path.join(self.log_dir))

    def stop_communication(self):
        self.cm.stop_iperf_communication()
        self.cm.close_kernel_communication()
        self.kernel_thread.exit()

    def _init_communication(self):
        # Start thread to communicate with kernel

        if not self.initiated:
            print("Start kernel thread...")

            # Thread for kernel info
            self.kernel_thread = KernelRequest(
                self.cm.netlink_communicator, self.num_fields_kernel)

            self.kernel_thread.start()

            print("Kernel thread started.")
            self.initiated = True

    def _read_data(self):
        kernel_info = self.kernel_thread.queue.get()
        self.kernel_thread.queue.task_done()
        return kernel_info
    
    def set_protocol(self):
        msg = self.cm.netlink_communicator.create_netlink_msg(
            'SENDING ACTION', msg_flags=2, msg_seq=self.proto_id)
        self.cm.netlink_communicator.send_msg(msg)
    
    def write_data(self, collected_data, path_to_file):
        data_string = ",".join([str(collected_data[key]) for key in collected_data])
        header = ",".join([str(key) for key in collected_data])
        with open(path_to_file, 'a') as f:
            # Check if the file is empty, and if so, write the header
            if f.tell() == 0:
                f.write(f"{header}\n")
            f.write(f"{data_string}\n")


    def run_collection(self):
        """ 
        TODO: we want to receive network parameters from the kernel side. In order to do that, we run a thread which is in charge of 
        communicating in real time with the kernel module. During the communication, the thread receive the "message" from the kernel 
        module, containing the network information, and store everything locally.
        """
        # Read zeta and kappa from config file
        with open('config/train.yml', 'r') as file:
            config = yaml.safe_load(file)

        # Check if the file already exists
        csv_dir = f'{self.log_dir}/csv'
        os.makedirs(csv_dir, exist_ok=True)
        path_to_file = os.path.join(csv_dir, f'{self.protocol}.bw{self.bw}.rtt{self.rtt}.bdp_mult{self.bdp_mult}.csv')
        if os.path.exists(path_to_file):
            # Empty the existing file (we overwrite the file with the new collection but same parameters)
            with open(path_to_file, 'w'):
                pass

        collected_data = {}
        start = time.time()
        self.set_protocol() # Communicate with kernel to set the protocol
        # reward_base = (thr - self.zeta * loss_rate) / rtt # rtt should never be 0 since min_rtt is set to be > 0
        # reward = pow(abs(reward_base), self.kappa)  # Adjust abs() if you want to consider the absolute value
        while time.time()-start < self.running_time:
            data = self._read_data()
            collected_data = {
            'now': data[0],
            'cwnd': data[1],
            'rtt': data[2],
            'rtt_dev': data[3],
            'rtt_min': data[4], 
            'MSS': data[5],     
            'delivered': data[6],
            'lost': data[7],
            'in_flight': data[8],
            'retransmitted': data[9],
            'selected_proto_id': data[10],
            'thruput': data[11]*1e-6, # bps -> Mbps
            'loss_rate': 0 if (data[7] + data[6]) == 0 else data[7] / (data[7] + data[6]),
            }
            
            reward = pow(abs(collected_data['thruput'] - config['reward']['zeta'] * collected_data['loss_rate'] / collected_data['rtt']), config['reward']['kappa'])
            collected_data['reward'] = reward
            
            print('\n')
            print('Reward:', reward)
            print('\n')
            
            print("-- Collected Data --")
            print("\n".join(f"{key}: {value}" for key, value in collected_data.items()))

            # Save collected data to csv file
            self.write_data(collected_data, path_to_file) 
