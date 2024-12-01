import pandas as pd
from sklearn.model_selection import train_test_split

data = pd.read_csv('./classifiers/data/result/result_0.csv')

# Seed
rand_seed = 100
train_test_seed = 101

# Get one hot encoding of columns B
one_hot = pd.get_dummies(data['ip_proto'], dtype=float, drop_first=True)
# Drop column B as it is now encoded
data = data.drop('ip_proto',axis = 1)
# Join the encoded df
data = data.join(one_hot)
# Reorder columns
data = data[['duration', 'UDP', 'TCP', 'out_packets', 'out_bytes','in_packets','in_bytes','label']]
# print(data)
# Shuffle rows
data = data.sample(frac=1, random_state=rand_seed).reset_index(drop=True)

X_train, X_test, y_train, y_test = train_test_split(data.iloc[:, :-1], data.iloc[:, -1], test_size=0.25, random_state=train_test_seed)

# print(data)
# print(X_train)
# print(y_train)
pd.concat([X_train, y_train], axis=1).to_csv('./classifiers/data/result/result_train_0.csv', index=False)
pd.concat([X_test, y_test], axis=1).to_csv('./classifiers/data/result/result_test_0.csv', index=False)