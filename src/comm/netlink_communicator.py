import os
from pickle import TRUE
import socket
import struct
import traceback
import errno
import time
import utilities.utils as utils

NETLINK_TEST = 25

class NetlinkCommunicator():
    _socket_obj = None

    def __init__(self):
        self.socket = self.create_socket()
        self.socket.setblocking(False)
        self.set_socket_buffer_size()

    # def init_socket(self):
    #     s = socket.socket(socket.AF_NETLINK, socket.SOCK_RAW, NETLINK_TEST)
    #     s.bind((os.getpid(), 0))
    #     return s
    
    @classmethod
    def create_socket(cls):
        if cls._socket_obj is None:
            s = socket.socket(socket.AF_NETLINK, socket.SOCK_RAW, NETLINK_TEST)
            s.bind((os.getpid(), 0))
            cls._socket_obj = s
        return cls._socket_obj

    def change_cca(self, protocol):
        msg = self.create_netlink_msg(
            'SENDING ACTION', msg_flags=2, msg_seq=protocol)
        self.send_msg(msg)

    def close_socket(self):
        self.socket.close()

    def create_netlink_msg(self, data, msg_type=0, msg_flags=0, msg_seq=0, msg_pid=os.getpid()):
        payload = f'{data}\0'
        header_size = 16
        payload_size = len(payload)
        msg_len = header_size + payload_size
        header = struct.pack("=LHHLL", msg_len, msg_type, msg_flags, msg_seq, msg_pid)
        msg = header + payload.encode()
        return msg

    def send_msg(self, msg):
        self.socket.send(msg)

    def set_socket_buffer_size(self, recv_size=10000, send_size=10000):
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, recv_size)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, send_size)

    def receive_msg(self):
        return self.socket.recv(8192)

    def read_netlink_msg(self, msg):
        value_len, value_type, value_flags, value_seq, value_pid = struct.unpack("=LHHLL", msg[:16])
        data = msg[16:value_len]
        return data
