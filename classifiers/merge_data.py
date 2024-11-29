import pandas as pd

udp_attack_1 = pd.read_csv('./classifiers/data/UDP_attack_1.csv')
udp_attack_2 = pd.read_csv('./classifiers/data/UDP_attack_2.csv')

udp_attack = pd.concat([udp_attack_1, udp_attack_2])
udp_attack.drop('Unnamed: 0', axis=1, inplace=True)
print(len(udp_attack))
udp_attack.to_csv('./classifiers/data/UDP_attack.csv', index=False)