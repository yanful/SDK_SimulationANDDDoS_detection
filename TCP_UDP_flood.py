import random
import socket
from multiprocessing import Process
import os
from argparse import ArgumentParser
import datetime


def flood(ip: str, udp: bool, port: int, times: int, timeout: int):
    data = random.randint(1, 1024).to_bytes(length=8, byteorder="big", signed=False)
    now = datetime.datetime.now()
    while True:
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM if udp else socket.SOCK_STREAM)
            addr = (str(ip), int(port))
            for _ in range(times):
                s.sendto(data, addr)
            print("Sent to " + str(port) + " process " + str(os.getpid()))
            if (datetime.datetime.now() - now).total_seconds() > timeout:
                break
        except Exception as err:
            print("Error " + str(err))


if __name__ == '__main__':

    parser = ArgumentParser("Simple UDP\\TCP flooder")
    parser.add_argument("--tcp", help="Use TCP (default: use UDP)", action="store_true")
    parser.add_argument("--port", help="Random port min", type=int, default=5001)
    parser.add_argument("--packets", help="Packets count per one connection", type=int, default=1000)
    parser.add_argument("--threads", help="Threads count", type=int, default=1)
    parser.add_argument("--timeout", help="Word time, sec", type=int, default=30)

    args = parser.parse_args()

    for _ in range(args.threads):
        Process(target=flood, args=("10.0.0.2", not args.tcp, args.port, args.packets,
                                    args.timeout)).start()