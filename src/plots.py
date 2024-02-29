import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import time
from argparse import ArgumentParser
import json
import utilities.utils as utils

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

def plot_training_reward_multi(timestamp, pool, exp, n_steps):
    """
    Plot the reward over time for a single training run
    """
    plt.figure(figsize=(20, 5))
    collection_dir = os.path.join(log_dir, 'collection', 'csv', n_steps)
    mutant_df = pd.read_csv(os.path.join(run_dir, 
        f"run.{timestamp}.csv"))
        
    for proto in pool:
        proto_df = pd.read_csv(os.path.join(collection_dir, 
            f"{proto}.bw{exp['bw']}.rtt{exp['rtt']}.bdp_mult{exp['bdp_mult']}.csv"))
        plt.plot(proto_df["reward"], label=proto)

    plt.plot(mutant_df["reward"][:len(proto_df["reward"])], label="mutant")
    plt.legend(fontsize=FONTSIZE)
    plt.xlabel("Steps", fontsize=FONTSIZE)
    plt.ylabel("Reward", fontsize=FONTSIZE)
    plt.xticks(fontsize=FONTSIZE)
    plt.yticks(fontsize=FONTSIZE)
    save_dir = os.path.join(plot_dir, ''.join(timestamp.split('.')))
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(f"{save_dir}/training_reward_over_time_multi.bw{exp['bw']}.rtt{exp['rtt']}.bdp_mult{exp['bdp_mult']}.png",
                bbox_inches='tight')

def plot_thr_single(timestamp, exp):
    """
    Plot a single statistic for a given protocol over time, given the experiment settings
    """
    df = pd.read_csv(os.path.join(run_dir, 
        f"run.{timestamp}.csv"))
    
    plt.figure(figsize=(20, 5))
    plt.plot(df["thruput"])
    plt.legend(fontsize=FONTSIZE)
    plt.xlabel("Steps", fontsize=FONTSIZE)
    plt.ylabel("Throughput [Mbps]", fontsize=FONTSIZE)
    plt.xticks(fontsize=FONTSIZE)
    plt.yticks(fontsize=FONTSIZE)
    save_dir = os.path.join(plot_dir, timestamp)
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(f"{save_dir}/mutant.throughput_over_time.bw{exp['bw']}.rtt{exp['rtt']}.bdp_mult{exp['bdp_mult']}.png",
                bbox_inches='tight')

def plot_thr_multi(timestamp, pool, exp, n_steps):
    """
    Plot a single statistic for multiple protocols over time
    """
    plt.figure(figsize=(20, 5))
    collection_dir = os.path.join(log_dir, 'collection', 'csv', n_steps)
    mutant_df = pd.read_csv(os.path.join(run_dir, 
        f"run.{timestamp}.csv"))
        
    for proto in pool:
        proto_df = pd.read_csv(os.path.join(collection_dir, 
            f"{proto}.bw{exp['bw']}.rtt{exp['rtt']}.bdp_mult{exp['bdp_mult']}.csv"))
        plt.plot(proto_df["thruput"], label=proto)

    plt.plot(mutant_df["thruput"][:len(proto_df["thruput"])], label="mutant")
    plt.legend(fontsize=FONTSIZE)
    plt.xlabel("Steps", fontsize=FONTSIZE)
    plt.ylabel("Throughput [Mbps]", fontsize=FONTSIZE)
    plt.xticks(fontsize=FONTSIZE)
    plt.yticks(fontsize=FONTSIZE)
    save_dir = os.path.join(plot_dir, ''.join(timestamp.split('.')))
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(f"{save_dir}/thruput_over_time_multi.bw{exp['bw']}.rtt{exp['rtt']}.bdp_mult{exp['bdp_mult']}.png",
                bbox_inches='tight')

def plot_rtt_single(timestamp, exp):
    """
    Plot a single statistic for a given protocol over time, given the experiment settings
    """
    df = pd.read_csv(os.path.join(run_dir, 
        f"run.{timestamp}.csv"))
    
    plt.figure(figsize=(20, 5))
    plt.plot(df["rtt"])
    plt.legend(fontsize=FONTSIZE)
    plt.xlabel("Steps", fontsize=FONTSIZE)
    plt.ylabel("RTT [ms]", fontsize=FONTSIZE)
    plt.xticks(fontsize=FONTSIZE)
    plt.yticks(fontsize=FONTSIZE)
    save_dir = os.path.join(plot_dir, timestamp)
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(f"{save_dir}/mutant.rtt_over_time.bw{exp['bw']}.rtt{exp['rtt']}.bdp_mult{exp['bdp_mult']}.png",
                bbox_inches='tight')
    
def plot_rtt_multi(timestamp, pool, exp, n_steps):
    """
    Plot a single statistic for multiple protocols over time
    """
    plt.figure(figsize=(20, 5))
    collection_dir = os.path.join(log_dir, 'collection', 'csv', n_steps)
    mutant_df = pd.read_csv(os.path.join(run_dir, 
        f"run.{timestamp}.csv"))
        
    for proto in pool:
        proto_df = pd.read_csv(os.path.join(collection_dir, 
            f"{proto}.bw{exp['bw']}.rtt{exp['rtt']}.bdp_mult{exp['bdp_mult']}.csv"))
        plt.plot(proto_df["rtt"], label=proto)

    plt.plot(mutant_df["rtt"][:len(proto_df["rtt"])], label="mutant")
    plt.legend(fontsize=FONTSIZE)
    plt.xlabel("Steps", fontsize=FONTSIZE)
    plt.ylabel("RTT [ms]", fontsize=FONTSIZE)
    plt.xticks(fontsize=FONTSIZE)
    plt.yticks(fontsize=FONTSIZE)
    save_dir = os.path.join(plot_dir, ''.join(timestamp.split('.')))
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(f"{save_dir}/rtt_over_time_multi.bw{exp['bw']}.rtt{exp['rtt']}.bdp_mult{exp['bdp_mult']}.png",
                bbox_inches='tight')

def plot_reward_with_actions():
    pass

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--timestamp', '-t', type=str)
    # parser.add_argument('--exp', '-e', type=str)
    # parser.add_argument('--proto', nargs='+', type=str, default=None, help='Protocols to compare with mutant')
    # parser.add_argument('--steps', '-s', type=int, default=None, help='Number of steps for multi plot')
    args = parser.parse_args()

    config = utils.parse_training_config()

    if args.timestamp:
        with open(os.path.join(log_dir, 'mab', 'settings.json'), 'r') as file:
            logs = [json.loads(log) for log in file.readlines()]
        index = next((i for i, log in enumerate(logs) if log.get("timestamp") == args.timestamp), None)
        if index is not None:
            run_log = logs[index]
            exp = {'bw': run_log['bw'], 'rtt': run_log['rtt'], 'bdp_mult': run_log['bdp_mult']}
            pool = run_log['action_mapping'].values()
            n_steps = str(run_log['num_steps'] * run_log['steps_per_loop'])
        else:
            raise ValueError(f"Timestamp {args.timestamp} not found in settings.json")
    else:
        raise ValueError("Timestamp not provided")

    plot_training_reward_single(args.timestamp, exp)
    plot_thr_single(args.timestamp, exp)
    plot_training_reward_multi(args.timestamp, pool, exp, n_steps)
    plot_thr_multi(args.timestamp, pool, exp, n_steps)
    plot_rtt_single(args.timestamp, exp)
    plot_rtt_multi(args.timestamp, pool, exp, n_steps)
