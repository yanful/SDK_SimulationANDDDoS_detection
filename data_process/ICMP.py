import csv
import re
import argparse

# Define server and client Ethernet addresses
SERVER_MAC = "9e:6e:30:89:34:4e"
CLIENT_MAC = "82:c5:23:c3:64:d4"

# Set up argument parser
parser = argparse.ArgumentParser(description="Extract OpenFlow flow data (ICMP) and save to a CSV file.")
parser.add_argument("-i", "--input", required=True, help="Path to the input file containing flow data.")
parser.add_argument("-o", "--output", required=True, help="Path to the output CSV file.")
args = parser.parse_args()

# Regular expressions for extracting fields
duration_pattern = r"duration=([\d.]+)s"
n_packets_pattern = r"n_packets=(\d+)"
n_bytes_pattern = r"n_bytes=(\d+)"
dl_src_pattern = r"dl_src=([0-9a-f:]+)"
dl_dst_pattern = r"dl_dst=([0-9a-f:]+)"

# Open the input and output files
with open(args.input, "r") as infile, open(args.output, "w", newline="") as outfile:
    # Initialize CSV writer
    csv_writer = csv.writer(outfile)
    csv_writer.writerow(["duration", "protocol", "in_packets", "in_bytes", "out_packets", "out_bytes"])
    
    # Skip the first two lines
    lines = infile.readlines()[2:]
    
    # Process data vectors (five-line blocks)
    for i in range(0, len(lines), 5):
        block = "".join(lines[i:i+3]).strip()

        if not block :
            continue
        
        # Extract fields from each flow
        matches = re.findall(r"cookie=.*?actions=.*?(?:\n|$)", block)
        if len(matches) < 2:
            continue
        
        # Extract data for each flow
        in_flow, out_flow = matches
        in_flow_dl_src = re.search(dl_src_pattern, in_flow).group(1)
        out_flow_dl_src = re.search(dl_src_pattern, out_flow).group(1)
        
        # Ensure the flows correspond to server and client
        if in_flow_dl_src == SERVER_MAC:
            incoming_flow, outgoing_flow = out_flow, in_flow
        else:
            incoming_flow, outgoing_flow = in_flow, out_flow
        
        # Extract flow metrics
        duration = re.search(duration_pattern, incoming_flow).group(1)
        protocol = "ICMP"  # Hardcoded as per the description
        in_packets = re.search(n_packets_pattern, incoming_flow).group(1)
        in_bytes = re.search(n_bytes_pattern, incoming_flow).group(1)
        out_packets = re.search(n_packets_pattern, outgoing_flow).group(1)
        out_bytes = re.search(n_bytes_pattern, outgoing_flow).group(1)
        
        # Write to CSV
        csv_writer.writerow([duration, protocol, in_packets, in_bytes, out_packets, out_bytes])

print(f"Data has been written to {args.output}")
