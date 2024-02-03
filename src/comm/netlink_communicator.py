import os
from pickle import TRUE
import socket
import struct
import traceback

NETLINK_TEST = 25

class NetlinkCommunicator():
    _socket_obj = None

    def __init__(self):
        self.socket = self.create_socket()

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

    def receive_msg(self):
        try:
            return self.socket.recv(8192)
        except Exception as err:
            print('\n')
            print(traceback.format_exc())
            self.socket.close()
            # clear buffer
            return None

    def read_netlink_msg(self, msg):
        value_len, value_type, value_flags, value_seq, value_pid = struct.unpack("=LHHLL", msg[:16])
        data = msg[16:value_len]
        return data
