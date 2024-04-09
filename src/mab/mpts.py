import numpy as np
from src.comm.kernel_thread import KernelRequest
from src.utilities import utils
from src.comm.netlink_communicator import NetlinkCommunicator
import time
import random 
from tqdm import tqdm

class MPTS:
    def __init__(self, arms: dict, k: int, T: int, step_wait: float, thread: KernelRequest, net_channel: NetlinkCommunicator):
        """
        Initializes the MPTS algorithm.

        Args:
            arms: A list or array of arms (e.g., your congestion control protocols).
            k: The desired number of top arms to select.
            T: The total budget (number of rounds).
        """
        self.arms = arms # mapping arms -> protocol id
        self.k = k
        if k > len(arms):
            raise ValueError("k cannot be greater than the number of arms.")
        self.T = T
        self.k_thread = thread # Thread to communicate with the kernel
        self.net_channel = net_channel
        self.proto_config = utils.parse_protocols_config() # for debug (protocol names)
        self.proto_names = {int(self.proto_config[p]['id']): p for p in self.proto_config.keys()}
        self.step_wait = step_wait # seconds
        # print("MPTS initialized with arms: ", self.arms, " k: ", self.k, " T: ", self.T, " step_wait: ", self.step_wait, " proto_names: ", self.proto_names)

    def compute_reward(self, kappa, zeta, thr, loss_rate, rtt):
        # Reward is normalized if the normalize_rw is true, otherwise max_rw = 1
        return (pow(abs((thr - zeta * loss_rate)), kappa) / (rtt*10**-3) )  # thr in Mbps; rtt in s
    
    def pull(self, arm):
        obs_list = []
        self.net_channel.change_cca(int(self.arms[arm]))
        self.k_thread.enable()
        start = time.time()
        # while time.time() - start < self.step_wait:
        while len(obs_list) < 10: # 5 samples
            obs = self.k_thread.read_data()
            obs_list.append(obs)
        self.k_thread.disable()
        self.k_thread.flush()
        # Average
        obs = np.mean(obs_list, axis=0)
        feature_settings = utils.parse_features_config()
        feature_names = feature_settings['kernel_info'] # List of features
        data = {feature: obs[i] for i, feature in enumerate(feature_names)}
        thr = data['thruput']*10**-6 # bps to Mbps
        loss_rate = data['loss_rate']*10**-2 # % to fraction
        rtt = data['rtt']*10**-3 # us to ms
        # print("Protocol: ", self.proto_names[int(self.arms[arm])], "thr: ", thr, " loss rate: ", loss_rate, " rtt: ", rtt, " cwnd", data['cwnd'])
        reward = self.compute_reward(kappa=2, zeta=5, thr=thr, loss_rate=loss_rate, rtt=rtt) # absolute value
        return reward

    # Helper functions for calculations
    def log_bar(self):  
        return 0.5 + sum(1 / i for i in range(2, self.T + 1))

    def compute_n_j(self, j, n):
        return int((self.T - n) / (n + 1 - j) * (1 / self.log_bar()))

    def run(self):
        """
        Implements the MPTS algorithm to select the best k arms.

        Args:
            arms: A list or array of arms (e.g., your congestion control protocols).
            k: The desired number of top arms to select.
            T: The total budget (number of rounds).

        Returns:
            A list containing the indices of the selected best 'k' arms.
        """

        n = len(self.arms)
        accepted = []  # Stores accepted arms
        active_arms = list(range(n))  # Indices of active arms
        k_remaining = self.k  

        # Phases of the algorithm
        print("Running MPTS algorithm...")
        for j in tqdm(range(n - 1)):
            print(f"Phase {j + 1}/{n - 1}")
            n_j = self.compute_n_j(j, n)
            arm_counts = np.zeros(n)  # Number of times each arm has been pulled
            arm_rewards = np.zeros(n)  # Sum of rewards for each arm

            # Pull a random arm from the active arms for n_j - n_{j-1} rounds
            # We randomize the selection to avoid dependencies between consecutive protocols
            for _ in tqdm(active_arms):
                for _ in range(n_j - (n_j - 1 if j > 0 else 0)): 
                    # Create a copy of active_arms to work with 
                    active_arms_copy = active_arms.copy()
                    while active_arms_copy:  
                        arm_index = random.choice(active_arms_copy)
                        reward = self.pull(arm_index)
                        # print("Arm ", self.arms[arm_index], " reward: ", reward)
                        arm_counts[arm_index] += 1
                        arm_rewards[arm_index] += reward

                        # Remove the used arm to prevent repetition in the next iteration
                        active_arms_copy.remove(arm_index) 

            # Calculate empirical means and gaps
            empirical_means = arm_rewards[active_arms] / arm_counts[active_arms]
            order = np.argsort(empirical_means)[::-1]  # Descending order
            # Print the empirical mean for each protocol
            # for i in order:
            #     print(f"Protocol {self.proto_names[int(self.arms[i])]}: {empirical_means[i]}")
            empirical_gaps = np.zeros(n - j)

            for r in range(n - j):
                if r <= k_remaining - 1:
                    empirical_gaps[r] = empirical_means[order[r]] - empirical_means[order[k_remaining]]
                else:
                    empirical_gaps[r] = empirical_means[order[k_remaining - 1]] - empirical_means[order[r]]

            # Deactivate and potentially accept an arm
            deactivated_index = np.argmax(empirical_gaps)
            deactivated_arm = active_arms[order[deactivated_index]]
            active_arms.remove(deactivated_arm)

            if empirical_means[order[deactivated_index]] > empirical_means[order[k_remaining]]:
                accepted.append(self.arms[deactivated_arm])
                k_remaining -= 1
        print("MPTS algorithm completed.")
        return accepted
