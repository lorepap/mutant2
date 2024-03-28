import numpy as np
from src.comm.kernel_thread import KernelRequest


class MPTS:
    def __init__(self, arms, k, T, thread):
        """
        Initializes the MPTS algorithm.

        Args:
            arms: A list or array of arms (e.g., your congestion control protocols).
            k: The desired number of top arms to select.
            T: The total budget (number of rounds).
        """
        self.arms = arms
        self.k = k
        self.T = T
        self.k_thread: KernelRequest = thread # Thread to communicate with the kernel

    def pull(self):
        self.k_thread.enable()
        self.k_thread.read_data()
        self.k_thread.disable()

    def mpts(self):
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

        # Helper functions for calculations
        def log_bar(T):  
            return 0.5 + sum(1 / i for i in range(2, T + 1))

        def compute_n_j(j, T, n):
            return int((T - n) / (n + 1 - j) * (1 / log_bar(T)))

        # Phases of the algorithm
        for j in range(n - 1):
            n_j = compute_n_j(j, T, n)
            arm_counts = np.zeros(n)  # Number of times each arm has been pulled
            arm_rewards = np.zeros(n)  # Sum of rewards for each arm

            # Pull active arms for n_j - n_{j-1} rounds
            for _ in range(n_j - (n_j - 1 if j > 0 else 0)): 
                for i in active_arms:
                    arm_index = i
                    reward = self.arms[arm_index].pull()  # Replace with your arm pulling logic
                    arm_counts[arm_index] += 1
                    arm_rewards[arm_index] += reward

            # Calculate empirical means and gaps
            empirical_means = arm_rewards[active_arms] / arm_counts[active_arms]
            order = np.argsort(empirical_means)[::-1]  # Descending order
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
                accepted.append(deactivated_arm)
                k_remaining -= 1

        return accepted

