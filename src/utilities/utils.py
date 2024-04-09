import os
import sys
import subprocess
import traceback
from datetime import datetime
import numpy as np
import yaml
import json
import logging

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import utilities.context as context
import re

TIME_FORMAT = '%Y.%m.%d.%H.%M.%S'
now = datetime.now()


def time_to_str() -> str:
    return now.strftime(TIME_FORMAT)

def str_to_time(time: str) -> datetime:
    return datetime.strptime(time, TIME_FORMAT)

def parse_protocols_config():
    with open(os.path.join(context.entry_dir, 'config/protocols.yml')) as config:
        return yaml.load(config, Loader=yaml.BaseLoader)

def parse_features_config():
    with open(os.path.join(context.entry_dir, 'config/features.yml')) as config:
        return yaml.load(config, Loader=yaml.BaseLoader)
    
def parse_mpts_config():
    with open(os.path.join(context.entry_dir, 'config/mpts.yml')) as config:
        return yaml.load(config, Loader=yaml.BaseLoader)

def parse_pantheon_protocols_config():
    location = get_fullpath('/pantheon/src/config.yml')

    if location == None or location.strip() == '':
         sys.exit('Pantheon not installed on your machine')

    with open(location) as config:
        return yaml.load(config, Loader=yaml.BaseLoader)
    
def parse_traces_config():
    with open(os.path.join(context.entry_dir, 'config/traces.yml')) as config:
        return yaml.load(config, Loader=yaml.BaseLoader)

def parse_training_config():
    with open(os.path.join(context.entry_dir, 'config/train.yml')) as config:
        return yaml.load(config, Loader=yaml.FullLoader)

def np_encoder(self, object):
        if isinstance(object, np.generic):
            return object.item()


def get_fullpath(file: str) -> str:

    try:

        path = subprocess.check_output(['locate', file])

        return path.strip().decode('utf-8')

    except Exception as _:
        print('\n')
        print(traceback.ormat_exc())


def change_file_name(old_name: str, new_name: str) -> None:
    try:
        os.rename(old_name, new_name)
    except Exception as _:
        print('\n')
        print(traceback.print_exc())


def check_dir(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path)

def get_private_ip():
    """
    Returns the private IP address of the host machine.
    """
    # Run the ifconfig command and capture the output
    output = subprocess.check_output(['ifconfig']).decode('utf-8')
    
    # Search the output for the private IP address using a regular expression
    # pattern = r'inet (192(?:\.\d{1,3}){2}\.\d{1,3})'
    pattern = r'inet (?:addr:)?(10\.0\.2\.15)'
    match = re.search(pattern, output)

    if match:
        # If a match is found, return the IP address
        ip_address = match.group(1)
    else:
        # If no match is found, set the IP address to None
        ip_address = None
    return ip_address

def extend_features_with_stats(all_features, stat_features):
    # all_feature does not need to be modified
    all_features = all_features.copy()
    # Iterate through train_stat_features
    for stat_feature in stat_features:
        for w_size in ['s', 'm', 'l']:
            # Check if the stat_feature is in all_features
            if stat_feature in all_features:
                # Append additional statistical features to all_features
                all_features.extend([f"{stat_feature}_{w_size}_avg", f"{stat_feature}_{w_size}_min", f"{stat_feature}_{w_size}_max"])
    return all_features

def get_training_features(all_features, stat_features, pool_size):
    # all_feature does not need to be modified
    all_features = [f for f in all_features.copy() if f not in 'crt_proto_id']
    # Iterate through train_stat_features
    for stat_feature in stat_features:
        for w_size in ['s', 'm', 'l']:
            # Check if the stat_feature is in all_features
            if stat_feature in all_features:
                # Append additional statistical features to all_features
                all_features.extend([f"{stat_feature}_{w_size}_avg", f"{stat_feature}_{w_size}_min", f"{stat_feature}_{w_size}_max"])
    # One hot encoding features. N_actions = 2**n_features, so n_features= log2(n_actions)
    all_features.extend([f"arm_{i}" for i in range(pool_size)])
    return all_features

def log_settings(filename: str, settings: dict, status: str = False, training_time: str = "") -> None:
    logger = logging.getLogger(__name__)  # Create a separate logger instance
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')
    file_handler = logging.FileHandler(filename)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)    
    settings['status'] = status
    if status == "success":
        settings['training_time'] = training_time
    log_message = json.dumps(settings)
    logger.info(log_message)

def update_log(filename, settings, status, training_time, global_step):
    with open(filename, 'r') as file:
        logs = [json.loads(log) for log in file.readlines()]

    timestamp = settings.get("timestamp")
    index = next((i for i, log in enumerate(logs) if log.get("timestamp") == timestamp), None)
    if index is not None:
        logs[index].update({"status": status, "global_step": global_step, "training_time": training_time})

    with open(filename, 'w') as file:
        file.writelines(json.dumps(log) + '\n' for log in logs)

def get_latest_ckpt_dir(settings):
    # Load settings.json
    filename = os.path.join(context.entry_dir, 'log/mab/settings.json')
    with open(filename, 'r') as file:
        logs = [json.loads(log) for log in file.readlines()]
    # Search for the latest entry with the same settings
    for log in reversed(logs):
        s_bw = settings.get("bw")
        s_rtt = settings.get("rtt")
        s_bdp_mult = settings.get("bdp_mult")
        s_bw_factor = settings.get("bw_factor")
        # s_actions = sorted([int(a) for a in settings.get("action_mapping").keys()])
        log_bw = log.get("bw")
        log_rtt = log.get("rtt")
        log_bdp_mult = log.get("bdp_mult")
        log_bw_factor = log.get("bw_factor")
        # log_actions = sorted([int(a) for a in log.get("action_mapping").keys()])
        
        if s_bw == log_bw and s_rtt == log_rtt and s_bdp_mult == log_bdp_mult and s_bw_factor == log_bw_factor:
            return log.get("checkpoint_dir")

def get_actions_from_experiment(timestamp):
    # Load settings.json
    filename = os.path.join(context.entry_dir, 'log/mab/settings.json')
    with open(filename, 'r') as file:
        logs = [json.loads(log) for log in file.readlines()]
    # Search for the latest entry with the same settings
    for log in logs:
        if log.get("timestamp") == timestamp:
            return log.get("action_mapping")