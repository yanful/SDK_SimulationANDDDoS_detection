import pandas as pd

path = './classifiers/data'
normal_datas = [f'{path}/normal/UDP_normal.csv', f'{path}/normal/TCP_normal.csv', f'{path}/normal/ICMP_normal.csv']
attack_datas = [f'{path}/attack/UDP_attack.csv', f'{path}/attack/TCP_attack.csv', f'{path}/attack/ICMP_attack.csv']

result = pd.DataFrame()
# Equal sample for normal and attack
for i in range(len(normal_datas)):
    norm = pd.read_csv(normal_datas[i])
    atck = pd.read_csv(attack_datas[i])
    # print(atck)
    result = pd.concat([result, norm.iloc[:len(atck), :], atck], axis=0)
    print(result)
result.to_csv('./classifiers/data/result/result_0.csv', index=False)