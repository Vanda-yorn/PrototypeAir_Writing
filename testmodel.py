# from trainmodel import*
import torch
# import torch.nn as nn
# import numpy as np
import pandas as pd
# import torch.optim as optim

from model import SimpleNet


# data_test1 = pd.read_csv('datasets/testdata2.csv', header=None).to_numpy()
# x_test1 = torch.tensor(data_test1[0][:-1], dtype=torch.float32)
# print(x_test1.size())
# y_test1 = torch.tensor(data_test1[:, 0], dtype=torch.long)
data = pd.read_csv('datasets/train_data.csv', header=None).to_numpy()
x = data[:, 1:]
y = data[:, 0]
y_test = data[:, 0]
# print(data.shape)

# y_test = torch.from_numpy(y_test)


m = x.shape[0]

in_sz = x.shape[1]
layers_sz = [1024, 512]
out_sz = 33


PATH = "save/savedModel"
net = SimpleNet(layers_sz, in_sz, out_sz)
net.load_state_dict(torch.load(PATH))


data_test = pd.read_csv('PredictData/scaledata3.csv', header=None).to_numpy()
# data_test.remove(data_test[0:][-1])
x_test = torch.tensor(data_test[0][:], dtype=torch.float32)
# y_test = torch.tensor(data_test[:, 0], dtype=torch.long)
# x_test = x_test.view(1, -1)
# print(y_test.size())

# print(data_test)
# x_test = torch.tensor(data_test[0][:-1], dtype=torch.float32)
# x_test = x_test.view(1, -1)  # Reshape input data to match input layer of neural network
# x_test = torch.from_numpy(x_test)
# print(y_test1)


net.eval()
z_test = net(x_test)

# print(z_test)

output = torch.argmax(z_test)
print(output)
# predict = torch.argmax(z_test, dim=1)
# print((predict==y_test1).sum())
# accuracy = (predict == y_test).to(torch.float).mean()
# print(accuracy)

# #MultiClass 
# # accuracy = (predict==y_test1).sum()/x_test1.size(0)
# # accuracy = (predict==y_test1).mean()

# print('Accuracy: %.2f' % (accuracy.item()*100))

# z_test = net(x_test)
# predict = torch.argmax(z_test, dim=1)
# # print(x_test.size(0))
# accuracy = (predict==y_test).sum()/x_test.size(0)

# print('Accuracy: %.2f' % (accuracy.item()*100))