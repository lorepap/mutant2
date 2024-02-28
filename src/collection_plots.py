import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import time
from argparse import ArgumentParser

sns.set_theme(style="darkgrid")

"""
Experiments as a dict: e.g., {"name": "Baseline", "bw": 42, "rtt": 20, "bdp_mult": 1}
"""

FONTSIZE = 18   

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
log_dir = os.path.join(root_dir, 'log')
plot_dir = os.path.join(root_dir, 'analysis', 'plots')
run_dir = os.path.join(log_dir, 'mab', 'run')
timestamp = time.strftime("%Y%m%d%H%M%S")

# Define the experiments based on provided parameters
exp_dict = {
    "Baseline": {"bw": 42, "rtt": 20, "bdp_mult": 1},
    "Low_Bandwidth": {"bw": 6, "rtt": 20, "bdp_mult": 1},
    "High_RTT": {"bw": 12, "rtt": 80, "bdp_mult": 1},
    "Large_Queue": {"bw": 12, "rtt": 20, "bdp_mult": 10},
    "Mixed_Conditions": {"bw": 42, "rtt": 30, "bdp_mult": 2},
    "Challenging_Network": {"bw": 6, "rtt": 100, "bdp_mult": 1},
    "Challenging_Network_2": {"bw": 12, "rtt": 30, "bdp_mult": 0.5},
}

def plot_thr_single(protocol, exp):
    """
    Plot a single statistic for a given protocol over time, given the experiment settings
    """
    df = pd.read_csv(os.path.join(collection_dir, 
        f"{protocol}.bw{exp['bw']}.rtt{exp['rtt']}.bdp_mult{exp['bdp_mult']}.csv"))
    
    plt.figure(figsize=(20, 5))
    plt.plot(df["thruput"], label=protocol)
    plt.legend(fontsize=FONTSIZE)
    plt.xlabel("Steps", fontsize=FONTSIZE)
    plt.ylabel("Throughput [Mbps]", fontsize=FONTSIZE)
    plt.xticks(fontsize=FONTSIZE)
    plt.yticks(fontsize=FONTSIZE)
    save_dir = os.path.join(plot_dir, timestamp)
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(f"{save_dir}/{protocol}.throughput_over_time.bw{exp['bw']}.rtt{exp['rtt']}.bdp_mult{exp['bdp_mult']}.png",
                bbox_inches='tight')

def plot_thr_multi(pool, exp):
    """
    Plot a single statistic for multiple protocols over time
    """
    plt.figure(figsize=(20, 5))
    for proto in pool:
        df = pd.read_csv(os.path.join(collection_dir,
            f"{proto}.bw{exp['bw']}.rtt{exp['rtt']}.bdp_mult{exp['bdp_mult']}.csv"))
        plt.plot(df["thruput"], label=proto)
    plt.legend(fontsize=FONTSIZE)
    plt.xlabel("Steps", fontsize=FONTSIZE)
    plt.ylabel("Throughput [Mbps]", fontsize=FONTSIZE)
    plt.xticks(fontsize=FONTSIZE)
    plt.yticks(fontsize=FONTSIZE)
    save_dir = os.path.join(plot_dir, timestamp)
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(f"{save_dir}/throughput_over_time.bw{exp['bw']}.rtt{exp['rtt']}.bdp_mult{exp['bdp_mult']}.png",
                bbox_inches='tight')

def plot_avg_thr_single(protocol, exp):
    """
    Plot the average of a single statistic for a given protocol over time, given the experiment settings
    """
    pass

def plot_rtt_single(protocol, exp):
    """
    Plot a single statistic for a given protocol over time, given the experiment settings
    """
    df = pd.read_csv(os.path.join(collection_dir, 
        f"{protocol}.bw{exp['bw']}.rtt{exp['rtt']}.bdp_mult{exp['bdp_mult']}.csv"))
    
    plt.figure(figsize=(20, 5))
    plt.plot(df["rtt"], label=protocol)
    plt.legend(fontsize=FONTSIZE)
    plt.xlabel("Steps", fontsize=FONTSIZE)
    plt.ylabel("RTT [ms]", fontsize=FONTSIZE)
    plt.xticks(fontsize=FONTSIZE)
    plt.yticks(fontsize=FONTSIZE)
    save_dir = os.path.join(plot_dir, timestamp)
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(f"{save_dir}/{protocol}.rtt_over_time.bw{exp['bw']}.rtt{exp['rtt']}.bdp_mult{exp['bdp_mult']}.png",
                bbox_inches='tight')

def plot_rtt_multi(pool, exp):
    """
    Plot a single statistic for multiple protocols over time
    """
    plt.figure(figsize=(20, 5))
    for proto in pool:
        df = pd.read_csv(os.path.join(collection_dir,
            f"{proto}.bw{exp['bw']}.rtt{exp['rtt']}.bdp_mult{exp['bdp_mult']}.csv"))
        plt.plot(df["rtt"], label=proto)
    plt.legend(fontsize=FONTSIZE)
    plt.xlabel("Steps", fontsize=FONTSIZE)
    plt.ylabel("RTT [ms]", fontsize=FONTSIZE)
    plt.xticks(fontsize=FONTSIZE)
    plt.yticks(fontsize=FONTSIZE)
    save_dir = os.path.join(plot_dir, timestamp)
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(f"{save_dir}/rtt_over_time.bw{exp['bw']}.rtt{exp['rtt']}.bdp_mult{exp['bdp_mult']}.png",
                bbox_inches='tight')

def plot_avg_rtt_single(protocol, exp):
    """
    Plot the average of a single statistic for a given protocol over time, given the experiment settings
    """
    pass

def plot_reward_single(protocol, exp):
    """
    Plot a single statistic for a given protocol over time, given the experiment settings
    """
    df = pd.read_csv(os.path.join(collection_dir, 
        f"{protocol}.bw{exp['bw']}.rtt{exp['rtt']}.bdp_mult{exp['bdp_mult']}.csv"))
    
    plt.figure(figsize=(20, 5))
    plt.plot(df["reward"], label=protocol)
    plt.legend(fontsize=FONTSIZE)
    plt.xlabel("Steps", fontsize=FONTSIZE)
    plt.ylabel("Reward", fontsize=FONTSIZE)
    plt.xticks(fontsize=FONTSIZE)
    plt.yticks(fontsize=FONTSIZE)
    save_dir = os.path.join(plot_dir, timestamp)
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(f"{save_dir}/{protocol}.reward_over_time.bw{exp['bw']}.rtt{exp['rtt']}.bdp_mult{exp['bdp_mult']}.png",
                bbox_inches='tight')

def plot_reward_multi(pool, exp):
    """
    Plot a single statistic for multiple protocols over time
    """
    plt.figure(figsize=(20, 5))
    for proto in pool:
        df = pd.read_csv(os.path.join(collection_dir, 
            f"{proto}.bw{exp['bw']}.rtt{exp['rtt']}.bdp_mult{exp['bdp_mult']}.csv"))
        plt.plot(df["reward"], label=proto)
    plt.legend(fontsize=FONTSIZE)
    plt.xlabel("Steps", fontsize=FONTSIZE)
    plt.ylabel("Reward", fontsize=FONTSIZE)
    plt.xticks(fontsize=FONTSIZE)
    plt.yticks(fontsize=FONTSIZE)
    save_dir = os.path.join(plot_dir, timestamp)
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(f"{save_dir}/reward_over_time.bw{exp['bw']}.rtt{exp['rtt']}.bdp_mult{exp['bdp_mult']}.png",
                bbox_inches='tight')
    

def plot_avg_reward_single(protocol, exp):
    """
    Plot the average of a single statistic for a given protocol over time, given the experiment settings
    """
    pass


if __name__ == "__main__":
    name = "Large_Queue"
    to_plot = exp_dict[name]

    parser = ArgumentParser()
    parser.add_argument("--proto", "-p", nargs='+', help='Protocols to plot')
    parser.add_argument("--collection_time", "-t", type=str, default="30s", help="Collection time")
    args = parser.parse_args()
    collection_dir = os.path.join(log_dir, 'collection', 'csv', args.collection_time)
    protos = args.proto
    for p in protos:
        plot_thr_single(p, to_plot)
        plot_rtt_single(p, to_plot)
        plot_reward_single(p, to_plot)
    if len(protos) > 1:
        plot_thr_multi(protos, to_plot)
        plot_rtt_multi(protos, to_plot)
        plot_reward_multi(protos, to_plot)