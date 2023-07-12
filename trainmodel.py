from model import *


data = pd.read_csv('datasets/train_data.csv', header=None).to_numpy()
x = data[:, 1:]
y = data[:, 0]
y_test = data[:, 0]
# print(data.shape)

# y_test = torch.from_numpy(y_test)


m = x.shape[0]

in_sz = x.shape[1]
print (in_sz)
# print(x.shape)
layers_sz = [1024, 512]
out_sz = 33
n_epoch = 100
b_sz = 200

net = SimpleNet(layers_sz, in_sz, out_sz)



opt = optim.Adam(net.parameters())
cost_func = nn.CrossEntropyLoss()

iter = 0
for epoch in range(n_epoch):
  i = 0
  p = np.random.permutation(m)
  x = x[p]
  y = y[p]
  while i<m:
    b_x = torch.tensor(x[i:i+b_sz], dtype=torch.float32)
    b_y = torch.tensor(y[i:i+b_sz], dtype=torch.long)

    b_z = net(b_x)
    cost = cost_func(b_z, b_y)
    
    if iter%20==0:
      print(epoch, iter, cost.item())

    cost.backward()
    opt.step()
    opt.zero_grad()

    i += b_sz
    iter += 1

# torch.save(model.state_dict(), f)
torch.save(net.state_dict(), "save/savedModel")
# torch.save(net, "save/savedModel")

# Load the saved model
# loaded_model = torch.load("save/savedModel")

# data_test = pd.read_csv('PredictData/scaledata3.csv', header=None).to_numpy()
# # data_test.remove(data_test[0:][-1])
# x_test = torch.tensor(data_test[0][:-1], dtype=torch.float32)


# net.eval()
# z_test = net(x_test)

# print(z_test)

# output = torch.argmax(z_test)
# print(output)

data_test = pd.read_csv('datasets/test_data1.csv', header=None).to_numpy()
x_test = torch.tensor(data_test[:, 1:], dtype=torch.float32)
y_test = torch.tensor(data_test[:, 0], dtype=torch.long)

z_test = net(x_test)
predict = torch.argmax(z_test, dim=1)
accuracy = (predict==y_test).sum()/x_test.size(0)

print('Accuracy: %.2f' % (accuracy.item()*100))