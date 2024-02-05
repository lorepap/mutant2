import os
import sys
import subprocess
import traceback
from utilities import context
from datetime import datetime
import numpy as np
import yaml
import json
import logging

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from comm.netlink_communicator import NetlinkCommunicator
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

def parse_prod_config():
    with open(os.path.join(context.entry_dir, 'config/prod.yml')) as config:
        return yaml.load(config, Loader=yaml.BaseLoader)

def np_encoder(self, object):
        if isinstance(object, np.generic):
            return object.item()

def get_number_of_actions(comm: NetlinkCommunicator) -> int:
    response = comm.receive_msg()
    data = comm.read_netlink_msg(response)
    data_decoded = data.decode('utf-8')
    split_data = data_decoded.split(';')

    nchoices = int(split_data[0])
    protocols = {}

    for index, entry in enumerate(split_data):
        if index > 0 and entry != '':
            protocolKey = entry.split(':')
            protocols[protocolKey[1]] = int(protocolKey[0])

    return nchoices, protocols


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

def log_settings(filename: str, settings: dict, status: str = False, training_time: str = "") -> None:
    logging.basicConfig(filename=filename, level=logging.INFO, format='%(message)s')
    settings['status'] = status
    if status == "success":
        settings['training_time'] = training_time
    log_message = json.dumps(settings)
    logging.info(log_message)

def update_log(filename, settings, status, training_time):
    with open(filename, 'r') as file:
        logs = [json.loads(log) for log in file.readlines()]

    timestamp = settings.get("timestamp")
    index = next((i for i, log in enumerate(logs) if log.get("timestamp") == timestamp), None)
    if index is not None:
        logs[index].update({"status": status, "training_time": training_time})

    with open(filename, 'w') as file:
        file.writelines(json.dumps(log) + '\n' for log in logs)
