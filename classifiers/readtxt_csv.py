import pandas as pd

# file_reader = open('./UDP_attack_sample.txt', 'r').read()
file_reader = open('./classifiers/data/flows_output_TCP_rand.txt', 'r').read()
# print(file_reader)
content = file_reader.split('\n')

# print(content)

ignore_idx = [i for i in range(0, len(content), 3)]
ignore_idx.pop(-1)
# print(ignore_idx)
# print(content)
# print(ignore_idx)

# Datapoint
duration = []
ip_proto = []
out_packets = []
out_bytes = []

in_packets = []
in_bytes = []

label = []

# print(content[117:])

for i in range(0, len(content), 3):

    # 0, 1, 2
    # 3, 4, 5
    # print(i)
    if i in ignore_idx:
        data_out_1 = content[i+1].strip().split(', ')
        data_in_1 = content[i+2].strip().split(', ')

        data_out_2 = content[i+4].strip().split(', ')
        data_in_2 = content[i+5].strip().split(', ')

        # IP
        ip_proto.append('TCP')

        # Duration
        dur_1 = data_out_1[1].replace('duration=','')
        dur_1 = float(dur_1.replace('s', ''))

        dur_2 = data_out_2[1].replace('duration=','')
        dur_2 = float(dur_2.replace('s', ''))

        # print(dur_1)
        dur = dur_2 - dur_1
        # print(dur_1, dur_2, dur)
        duration.append(dur)

        # Out packets
        out_p_1 = int(data_out_1[3].replace('n_packets=', ''))
        out_p_2 = int(data_out_2[3].replace('n_packets=', ''))
        out_p = out_p_2 - out_p_1
        # print(out_p_1, out_p_2, out_p)
        out_packets.append(out_p)

        # Out bytes
        out_b_1 = int(data_out_1[4].replace('n_bytes=', ''))
        out_b_2 = int(data_out_2[4].replace('n_bytes=', ''))
        out_b = out_b_2 - out_b_1
        # print(out_b_1, out_b_2, out_b)
        out_bytes.append(out_b)

        # In packets
        in_p_1 = int(data_in_1[3].replace('n_packets=', ''))
        in_p_2 = int(data_in_2[3].replace('n_packets=', ''))
        in_p = in_p_2 - in_p_1
        # print(in_p_1, in_p_2, in_p)
        in_packets.append(in_p)

        # Out bytes
        in_b_1 = int(data_in_1[4].replace('n_bytes=', ''))
        in_b_2 = int(data_in_2[4].replace('n_bytes=', ''))
        in_b = in_b_2 - in_b_1
        # print(in_b_1, in_b_2, in_b)
        in_bytes.append(in_b)

        # print(dur_1)
        label.append(1)
    data = pd.DataFrame({'duration': duration, 'ip_proto': ip_proto, 
                         'out_packets': out_packets, 'out_bytes': out_bytes,
                        'in_packets': in_packets, 'in_bytes': in_bytes, 'label': label})
    # print(i)
    # print(data)
    data.to_csv('./classifiers/data/TCP_attack.csv')
    # print(dur_2)
