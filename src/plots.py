import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import time
from argparse import ArgumentParser

sns.set_style("darkgrid")

FONTSIZE = 18

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
log_dir = os.path.join(root_dir, 'log')
plot_dir = os.path.join(root_dir, 'analysis', 'plots')
run_dir = os.path.join(log_dir, 'mab', 'run')

experiments = {
    "Baseline": {"bw": 42, "rtt": 20, "bdp_mult": 1},
    "Low_Bandwidth": {"bw": 6, "rtt": 20, "bdp_mult": 1},
    "High_RTT": {"bw": 12, "rtt": 80, "bdp_mult": 1},
    "Large_Queue": {"bw": 12, "rtt": 20, "bdp_mult": 10},
    "Mixed_Conditions": {"bw": 42, "rtt": 30, "bdp_mult": 2},
    "Challenging_Network": {"bw": 6, "rtt": 100, "bdp_mult": 1},
    "Challenging_Network_2": {"bw": 12, "rtt": 30, "bdp_mult": 0.5},
}


def plot_training_reward_single(timestamp, exp):
    """
    Plot the reward over time for a single training run
    """
    df = pd.read_csv(os.path.join(run_dir, 
        f"run.{timestamp}.csv"))
    
    plt.figure(figsize=(20, 5))
    plt.plot(df["reward"])
    # plt.legend(fontsize=FONTSIZE)
    plt.xlabel("Steps", fontsize=FONTSIZE)
    plt.ylabel("Reward", fontsize=FONTSIZE)
    plt.xticks(fontsize=FONTSIZE)
    plt.yticks(fontsize=FONTSIZE)
    save_dir = os.path.join(plot_dir, ''.join(timestamp.split('.')))
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(f"{save_dir}/training_reward_over_time.bw{exp['bw']}.rtt{exp['rtt']}.bdp_mult{exp['bdp_mult']}.png",
                bbox_inches='tight')


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--timestamp', '-t', type=str)
    parser.add_argument('--exp', '-e', type=str)
    args = parser.parse_args()

    plot_training_reward_single(args.timestamp, experiments[args.exp])
