#!/bin/bash

# Define the output file
OUTPUT_FILE="flows_output_UDP_rand.txt"

# Ensure the output file exists and is writable
touch "$OUTPUT_FILE"

# Loop to run the command every 0.1 second
while true; do
    # Run the command and get the second line of the output
    SECOND_LINE=$(sudo ovs-ofctl -O OpenFlow13 dump-flows s1 | sed -n '3p')
    # SECOND_LINE=$(sudo ovs-ofctl -O OpenFlow13 dump-flows s1 );

    
    # Append the second line to the output file
    echo "$SECOND_LINE" >> "$OUTPUT_FILE";
    
    # Wait for 30 second
    sleep 30;
done

# h1: 22:40:b4