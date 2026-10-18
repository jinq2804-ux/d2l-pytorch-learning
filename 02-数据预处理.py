import os
import pandas as pd
import torch

os.makedirs(os.path.join('..', 'data'), exist_ok=True)
data_file = os.path.join('..', 'data', 'house_tiny.csv')
with open(data_file, 'w') as f:
    f.write('NumRooms,Alley,Price\n') #列名
    f.write('NA,Pave,12500\n') #每行表示一个数据样本
    f.write('2,NA,10600\n')
    f.write('4,OL,17800\n')
    f.write('NA,NA,140000\n')

data = pd.read_csv(data_file)
print(data)

inputs,outputs = data.iloc[:,0:2], data.iloc[:,2]
inputs = inputs.fillna(inputs.mean(numeric_only=True))
print(inputs)

#对于 inputs 中的类别值和离散值，我们将 NaN 视为一个类别
inputs = pd.get_dummies(inputs,columns=['Alley'],dummy_na=True)
print(inputs)

#现在input和output所有条目都是数据类型，可以转换为张量格式
x, y =torch.tensor(inputs.astype('float64').values), torch.tensor(outputs.values)
print(x,y)