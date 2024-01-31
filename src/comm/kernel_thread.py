import queue
import threading
import traceback
import select

from comm.netlink_communicator import NetlinkCommunicator


class KernelRequest(threading.Thread):
    def __init__(self, comm: NetlinkCommunicator, num_fields_kernel: int):
        threading.Thread.__init__(self)
        self.comm = comm
        self.num_fields_kernel = num_fields_kernel
        self.queue = queue.Queue()
        self.exit_event = threading.Event()  # Event to signal thread to exit

    def run(self):
        while True:
            try:

                if self.comm.socket.fileno() == -1:
                    # Invalid file descriptor, break out of the loop
                    print("[KERNEL THREAD] Invalid file descriptor. Exiting...")
                    break
                
                # Use select with a timeout to implement a timer
                readable, _, _ = select.select([self.comm.socket], [], [], 10.0)  # Set timeout to 5 seconds

                if not readable:
                    # No data received within the timeout period
                    print("[KERNEL THREAD] Timeout occurred. Exiting...")
                    break
                
                # print("[KERNEL THREAD] Waiting for message...")
                msg = self.comm.receive_msg()
                # print("[KERNEL THREAD] Received message:", msg)
                if msg:
                    data = self.comm.read_netlink_msg(msg)
                    # print("[KERNEL THREAD] Received data:", data)
                    data_decoded = data.decode('utf-8')
                    if data_decoded == "0":
                        # Received "0" as a notification of completed setup
                        print("[KERNEL THREAD] Communication setup completed.")
                    elif data_decoded == "-1":
                        print("[KERNEL THREAD] Communication terminated")
                        break
                    else:
                        split_data = data_decoded.split(';')[:self.num_fields_kernel]
                        entry = list(map(int, split_data))
                        # print("[KERNEL THREAD] Data received:", entry)
                        self.queue.put(entry)
                        # print("[KERNEL THREAD] Queue contents:", list(self.queue.queue))
                else:
                    print("[KERNEL THREAD] Exit event set. Exiting...")
                    break
            except Exception as _:
                print('\n')
                print(traceback.format_exc())

    def exit(self):
        print("[KERNEL THREAD] Exiting...")
        self.exit_event.set()  # Set the exit event to signal thread to exit
