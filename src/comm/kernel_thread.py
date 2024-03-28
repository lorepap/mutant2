import queue
import threading
import traceback
import select

from src.comm.netlink_communicator import NetlinkCommunicator


class KernelRequest(threading.Thread):
    def __init__(self, comm: NetlinkCommunicator, num_fields_kernel: int):
        threading.Thread.__init__(self)
        self.comm = comm
        self.num_fields_kernel = num_fields_kernel
        self.queue = queue.Queue()
        self.exit_event = threading.Event()  # Event to signal thread to exit
        self.enabled = False

    def enable(self):
        self.enabled = True

    def disable(self):
        self.enabled = False

    def flush(self):
        while not self.queue.empty():
            self.queue.get()

    def read_data(self):
        kernel_info = self.queue.get()
        # self.kernel_thread.queue.task_done()
        return kernel_info

    def run(self):
        while True:
            try:

                if self.comm.socket.fileno() == -1:
                    # Invalid file descriptor, break out of the loop
                    print("[KERNEL THREAD] Invalid file descriptor. Exiting...")
                    break
                
                # Use select with a timeout to implement a timer
                readable, _, _ = select.select([self.comm.socket], [], [], 20) 

                if not readable:
                    # No data received within the timeout period
                    print("[KERNEL THREAD] Timeout occurred. Exiting...")
                    break
                
                # print("[KERNEL THREAreturn self.socket.return self.socket.recv(8192)recv(8192)D] Waiting for message...")
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
                        split_data = data_decoded.split(';')
                        entry = [int(field) if field.isdigit() or (field[1:].isdigit() and field[0] == '-') else field for field in split_data]
                        # print("[KERNEL THREAD] Data received:", entry)
                        if self.enabled:
                            self.queue.put(entry)
                        # print queue size
                        # print("[KERNEL THREAD] Queue size:", self.queue.qsize())
                        # print("[KERNEL THREAD] Queue contents:", list(self.queue.queue))
                else:
                    print("[KERNEL THREAD] Exit event set. Exiting...")
                    break
            except Exception as _:
                print('\n')

    def exit(self):
        print("[KERNEL THREAD] Exiting...")
        self.exit_event.set()  # Set the exit event to signal thread to exit
