import re
import csv
import argparse

# Argument parser setup
parser = argparse.ArgumentParser(description="Extract data vectors and convert to CSV.")
parser.add_argument("-i", "--input", required=True, help="Path to the input file.")
parser.add_argument("-o", "--output", required=True, help="Path to the output CSV file.")
args = parser.parse_args()

# Regular expressions for extracting the fields
duration_pattern = r"duration=(\d+\.\d+)"
packets_pattern = r"n_packets=(\d+)"
bytes_pattern = r"n_bytes=(\d+)"

# Open the input file and create the output CSV file
with open(args.input, "r") as infile, open(args.output, "w", newline="") as outfile:
    # Initialize CSV writer
    csv_writer = csv.writer(outfile)
    
    # Write the CSV header
    csv_writer.writerow(["flow_duration", "n_packets", "n_bytes"])
    
    # Process each line in the input file
    for line in infile:
        # Extract the fields using regex
        flow_duration = re.search(duration_pattern, line)
        n_packets = re.search(packets_pattern, line)
        n_bytes = re.search(bytes_pattern, line)
        
        # If all fields are found, write them to the CSV
        if flow_duration and n_packets and n_bytes:
            csv_writer.writerow([
                flow_duration.group(1),
                n_packets.group(1),
                n_bytes.group(1)
            ])

print(f"CSV file has been created: {args.output}")
