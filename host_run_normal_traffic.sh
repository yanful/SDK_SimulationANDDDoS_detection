#! /bin/bash

# Define the output file
OUTPUT_FILE="flows_output.txt"

# Ensure the output file exists and is writable
touch "$OUTPUT_FILE"

# for i in $(seq 1 20);do
#     echo "Iteration $i"
#     # iperf -u -c 10.0.0.1  -t 1;
#     iperf -c 10.0.0.1  -t 0.5;
#     sleep 1;
# done


while true;do
    RANDOM_SLEEP=$(awk -v min=0.1 -v max=1.5 'BEGIN{srand(); print min+rand()*(max-min)}')

    # iperf -u -c 10.0.0.1  -t 1;
    iperf  -u -c 10.0.0.1  -t $RANDOM_SLEEP;
    sleep $RANDOM_SLEEP;
    # echo "Iteration $RANDOM_SLEEP"
done