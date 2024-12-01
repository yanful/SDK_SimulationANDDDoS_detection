import pandas as pd

def process_data(data, data_name):
    new_data = pd.DataFrame()
    for i in range(len(data)-1):
        new_point = data.iloc[i+1, [0, 2, 3, 4, 5]] - data.iloc[i, [0, 2, 3, 4, 5]]
        new_data = pd.concat([new_data, new_point.T], axis=1)
    new_data = new_data.T.reset_index(drop=True)
    ip_proto = data['protocol'].values[0]
    new_data['ip_proto'] = [ip_proto for _ in range(len(new_data))]
    new_data['label'] = [0 for _ in range(len(new_data))]

    new_data = new_data[['duration','ip_proto','out_packets','out_bytes','in_packets','in_bytes','label']]
    new_data.to_csv(f'./classifiers/data/normal/{data_name}.csv', index=False)  

UDP_data = pd.read_csv('./classifiers/data/normal/UDP_normal_flow.csv')
TCP_data = pd.read_csv('./classifiers/data/normal/TCP_normal_flow.csv')
ICMP_data = pd.read_csv('./classifiers/data/normal/ICMP_normal_flow.csv')

# print(UDP_data.iloc[1, 1])
# print(UDP_data.iloc[1, [0, 2, 3, 4, 5]] - UDP_data.iloc[0, [0, 2, 3, 4, 5]])
process_data(UDP_data, 'UDP_normal')
process_data(TCP_data, 'TCP_normal')
process_data(ICMP_data, 'ICMP_normal')