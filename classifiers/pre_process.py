import numpy as np

x_datas = ['./data/UDP.csv'] 
y_normal_data = ['./data/normal.csv']
y_attack_data = ['./data/attack.csv']

normal_data = np.array([])

# Normal
for x_data in x_datas:
    x = np.loadtxt(x_data, delimited=',')
    # for y_data in y_datas:
    y = np.loadtxt(y_normal_data, delimited=',')
    # Assuming x and y on the same length
    data = np.concatenate((x, y), axis=1)

    normal_data = np.concatenate((normal_data, data), axis=0)

attack_data = np.array([])

# Attack
for x_data in x_datas:
    x = np.loadtxt(x_data, delimited=',')
    # for y_data in y_datas:
    y = np.loadtxt(y_attack_data, delimited=',')
    # Assuming x and y on the same length
    data = np.concatenate((x, y), axis=1) # 

    attack_data = np.concatenate((attack_data, data), axis=0)

# Concatenate
result = np.concatenate((normal_data, attack_data), axis=0)

# Shuffle the array
seed = 0
np.random.shuffle(result, seed=0)

np.savetxt("./data/result.csv", result, delimiter=',')
    