# import subprocess

# output = subprocess.check_output(['python', 'Pose Classification/Classifier.py'])
# print(output.decode('utf-8'))

# This code mainly divides the data into training data and test data

import random
import shutil
import os

data_path = './image_to_detect'
output_train = './data/train'
output_test = './data/test'
data1 = '/angles'
data2 = '/norm'
try:
    os.mkdir('./data')
    os.mkdir(output_train)
    os.mkdir(output_test)
    os.mkdir(output_train+data1)
    os.mkdir(output_train+data2)
    os.mkdir(output_test+data1)
    os.mkdir(output_test+data2)
except:
    print('dir already exists')
    pass
list = []
for i in os.listdir(data_path+data1):
    list.append(os.path.join(data_path,data1,i))
random.shuffle(list)
length = int(len(list)*0.7)
for ii in list[:length]:
    # print(ii)
    shutil.copy(data_path+ii,output_train+ii)
for ii in list[length:]:
    # print(ii)
    shutil.copy(data_path+ii,output_test+ii)

list = []
for j in os.listdir(data_path+data2):
    list.append(os.path.join(data_path,data2,j))
random.shuffle(list)
for ii in list[:length]:
    # print(ii)
    shutil.copy(data_path+ii,output_train+ii)
for ii in list[length:]:
    # print(ii)
    shutil.copy(data_path+ii,output_test+ii)
# print(list)

# The models are trained using the training set (two models) and the two trained models are saved to the current directory.

import os
import numpy
import numpy as np
import torch
import torch.nn.functional as F

from torch import nn, optim
import os
#
data_path = './data'


data1 = 'angles'
data2 = 'norm'




def data_read(path,data_type):
    # global data1,data2
    max_len = 0
    if data_type == data1:
        divid = 1
    elif data_type == data2:
        divid = 8

    for i in os.listdir(os.path.join(path+'/train',data_type)):
        l = np.array(numpy.loadtxt(os.path.join(path+'/train', data_type,i)), dtype=float)
        leng = l.shape[0]
        if leng>max_len:
            max_len=leng

    label_dict={}
    idx=0
    train_x = []
    train_y = []
    for i in os.listdir(os.path.join(path + '/train', data_type)):
        l = np.array(numpy.loadtxt(os.path.join(path + '/train', data_type, i)), dtype=float)
        l = np.nan_to_num(l)
        if not np.any(l):
            # print('gg')
            continue
        fb, _, _, lb, _ = i.split('_')

        if 'front' in fb:
            # l = numpy.loadtxt(os.path.join(path+data_type,t))
            # print(l)
            if data_type == data1:
                split_line = 1
                if l.shape[0] < max_len:
                    # print('old l ', l, l.shape)
                    l = np.pad(l, ((0, max_len - l.shape[0]), (0, 0)), 'constant', constant_values=(0, 0))
                    # print('new l ', l, l.shape)
                train_x.append(l)

                if lb not in label_dict:
                    label_dict[lb] = idx
                    idx += 1
                train_y.append(label_dict[lb])

            if data_type == data2:
                split_line = 8
                # print('la', l.shape)
                if l.shape[0]<max_len:
                    l = np.pad(l, ((0, max_len - l.shape[0]), (0, 0)), 'constant', constant_values=(0, 0))
                    # print('lb', l, l.shape)

                l = l.reshape([-1,split_line,3])

                train_x.append(l)
                if lb not in label_dict:
                    label_dict[lb] = idx
                    idx += 1
                train_y.append(label_dict[lb])


                # print('lc',l.shape)
                # aa
                # if l.shape[0]<max_len:
                #     l = np.pad(l, ((0, max_len - l.shape[0]), (0, 0)), 'constant', constant_values=(0, 0))
                #     print('lb',l,l.shape)
                #     aa
        elif 'back' in fb:

            if data_type == data1:
                split_line = 1
                # print('l b',l)
                l = l[:,[3,2,1,0]]
                # print('l f', l)
                # aa
                if l.shape[0] < max_len:
                    # print('old l ', l, l.shape)
                    l = np.pad(l, ((0, max_len - l.shape[0]), (0, 0)), 'constant', constant_values=(0, 0))
                    # print('new l ', l, l.shape)
                train_x.append(l)

                if lb not in label_dict:
                    label_dict[lb] = idx
                    idx += 1
                train_y.append(label_dict[lb])


                # print(l,nl)
            elif data_type == data2:
                split_line = 8
                nl = l.reshape([-1,split_line,3])
                # print(nl.shape)
                # aa
                split_line = 8
                # print('la', l.shape)
                if l.shape[0] < max_len:
                    l = np.pad(l, ((0, max_len - l.shape[0]), (0, 0)), 'constant', constant_values=(0, 0))
                    # print('lb', l, l.shape)

                l = l.reshape([-1, split_line, 3])

                l = l[:, [5,4,3, 2, 1, 0,7,6],:]
                # print(l.shape)


                train_x.append(l)
                if lb not in label_dict:
                    label_dict[lb] = idx
                    idx += 1
                train_y.append(label_dict[lb])

    train_x = numpy.array(train_x)
    train_y = numpy.array(train_y)

    test_x = []
    test_y = []
    for i in os.listdir(os.path.join(path + '/test', data_type)):
        l = np.array(numpy.loadtxt(os.path.join(path + '/test', data_type, i)), dtype=float)
        l = np.nan_to_num(l)
        if not np.any(l):
            # print('gg')
            continue
        fb, _, _, lb, _ = i.split('_')

        if 'front' in fb:
            # l = numpy.loadtxt(os.path.join(path+data_type,t))
            # print(l)
            if data_type == data1:
                split_line = 1
                if l.shape[0] < max_len:
                    # print('old l ', l, l.shape)
                    l = np.pad(l, ((0, max_len - l.shape[0]), (0, 0)), 'constant', constant_values=(0, 0))
                    # print('new l ', l, l.shape)
                test_x.append(l)

                if lb not in label_dict:
                    label_dict[lb] = idx
                    idx += 1
                test_y.append(label_dict[lb])

            if data_type == data2:
                split_line = 8
                # print('la', l.shape)
                if l.shape[0] < max_len:
                    l = np.pad(l, ((0, max_len - l.shape[0]), (0, 0)), 'constant', constant_values=(0, 0))
                    # print('lb', l, l.shape)

                l = l.reshape([-1, split_line, 3])

                test_x.append(l)
                if lb not in label_dict:
                    label_dict[lb] = idx
                    idx += 1
                test_y.append(label_dict[lb])

                # print('lc',l.shape)
                # aa
                # if l.shape[0]<max_len:
                #     l = np.pad(l, ((0, max_len - l.shape[0]), (0, 0)), 'constant', constant_values=(0, 0))
                #     print('lb',l,l.shape)
                #     aa
        elif 'back' in fb:

            if data_type == data1:
                split_line = 1
                # print('l b',l)
                l = l[:, [3, 2, 1, 0]]
                # print('l f', l)
                # aa
                if l.shape[0] < max_len:
                    # print('old l ', l, l.shape)
                    l = np.pad(l, ((0, max_len - l.shape[0]), (0, 0)), 'constant', constant_values=(0, 0))
                    # print('new l ', l, l.shape)
                test_x.append(l)

                if lb not in label_dict:
                    label_dict[lb] = idx
                    idx += 1
                test_y.append(label_dict[lb])

                # print(l,nl)
            elif data_type == data2:
                split_line = 8
                nl = l.reshape([-1, split_line, 3])
                # print(nl.shape)
                # aa
                split_line = 8
                # print('la', l.shape)
                if l.shape[0] < max_len:
                    l = np.pad(l, ((0, max_len - l.shape[0]), (0, 0)), 'constant', constant_values=(0, 0))
                    # print('lb', l, l.shape)

                l = l.reshape([-1, split_line, 3])

                l = l[:, [5, 4, 3, 2, 1, 0, 7, 6], :]
                # print(l.shape)

                test_x.append(l)
                if lb not in label_dict:
                    label_dict[lb] = idx
                    idx += 1
                test_y.append(label_dict[lb])

    test_x = numpy.array(test_x)
    test_y = numpy.array(test_y)



    # print(train_x.shape,train_y.shape)
    # print(test_x.shape,test_y.shape)
    # print(i,data_type,label_dict)
    return train_x,train_y,test_x,test_y,label_dict


class FCN_Net(nn.Module):
    """
    Based on the simpleNet above, activation function is added to the output section of each layer
    """

    def __init__(self, in_dim, out_dim):
        super(FCN_Net, self).__init__()
        self.layer1 = nn.Sequential(nn.Linear(in_dim, 300), nn.ReLU(True))
        self.layer2 = nn.Sequential(nn.Linear(300, 512), nn.ReLU(True))
        self.layer3 = nn.Sequential(nn.Linear(512, 1024), nn.ReLU(True))
        self.layer4 = nn.Sequential(nn.Linear(1024, 300), nn.ReLU(True))
        self.layer5 = nn.Sequential(nn.Linear(300, out_dim))
        # self.layer6 =nn.sigmoid
        """
        The function of the Sequential() function here is to combine the layers of the network together.
        """

    def forward(self, x):
        # print(x.shape)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        # x = F.sigmoid(x)
        return x

import torch.utils.data as Data
train_x,train_y,test_x,test_y,label_dict = data_read(data_path,data1)
train_x =train_x.reshape([-1,train_x.shape[1]*train_x.shape[2]])
np.save('./'+data1+'_dict.npy',label_dict,allow_pickle=True)
# print('type: ',torch.LongTensor(train_y))
train_y = F.one_hot(torch.tensor(train_y).to(torch.int64),len(label_dict))
print('model training : ',data1)
print('training data shape: ' ,train_x.shape,train_y.shape)
print('test data shape: ' ,test_x.shape,test_y.shape)


train_dataset = Data.TensorDataset(torch.tensor(train_x,dtype=torch.float32), torch.tensor(train_y,dtype=torch.float32))
train_loader = Data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_dataset = Data.TensorDataset(torch.tensor(test_x,dtype=torch.float32), torch.tensor(test_y,dtype=torch.float32))
test_loader = Data.DataLoader(test_dataset, batch_size=64, shuffle=True)
model = FCN_Net(train_x.shape[1], len(label_dict))
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.02)
# print(model)
model.train()
acc_M=0
for i in range(200):
    correct = 0.0
    num = 0.0
    for batch_x, batch_y in train_loader:
        # print(batch_x,batch_y)
        pred = model(batch_x)

        loss = criterion(pred, batch_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pred = pred.max(1)[1]
        batch_y = batch_y.max(1)[1]

        # print(pred, batch_y.shape)
        correct += (pred == batch_y).sum()
        num += len(batch_y)
    print("epoch:", i + 1)
    # print('loss=', '{:.6f}'.format(loss.item()))
    acc = correct.item() / num
    print(f'loss: {loss.item():.4f}  | acc: {acc * 100:.2f}%')
    if acc > acc_M:
        acc_M = acc
        torch.save(model.state_dict(), './'+data1+'_model.pkl')







train_x,train_y,test_x,test_y,label_dict = data_read(data_path,data2)
train_x =train_x.reshape([-1,train_x.shape[1]*train_x.shape[2]*train_x.shape[3]])
np.save('./'+data2+'_dict.npy',label_dict,allow_pickle=True)
# print('type: ',torch.LongTensor(train_y))
train_y = F.one_hot(torch.tensor(train_y).to(torch.int64),len(label_dict))
print('model training : ',data2)
print('training data shape: ' ,train_x.shape,train_y.shape)
print('test data shape: ' ,test_x.shape,test_y.shape)


train_dataset = Data.TensorDataset(torch.tensor(train_x,dtype=torch.float32), torch.tensor(train_y,dtype=torch.float32))
train_loader = Data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_dataset = Data.TensorDataset(torch.tensor(test_x,dtype=torch.float32), torch.tensor(test_y,dtype=torch.float32))
test_loader = Data.DataLoader(test_dataset, batch_size=64, shuffle=True)
model = FCN_Net(train_x.shape[1], len(label_dict))
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.02)
# print(model)
model.train()
acc_M=0
for i in range(200):
    correct = 0.0
    num = 0.0
    for batch_x, batch_y in train_loader:
        # print(batch_x,batch_y)
        pred = model(batch_x)

        loss = criterion(pred, batch_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pred = pred.max(1)[1]
        batch_y = batch_y.max(1)[1]

        # print(pred, batch_y.shape)
        correct += (pred == batch_y).sum()
        num += len(batch_y)
    print("epoch:", i + 1)
    # print('loss=', '{:.6f}'.format(loss.item()))
    acc = correct.item() / num
    print(f'loss: {loss.item():.4f}  | acc: {acc * 100:.2f}%')
    if acc > acc_M:
        acc_M = acc
        torch.save(model.state_dict(), './'+data2+'_model.pkl')

# Tests were carried out using a test set and the accuracy of the two models were compared.

import os
import numpy
import numpy as np
import torch
import torch.nn.functional as F

from torch import nn, optim
import os
#
data_path = './data'


data1 = 'angles'
data2 = 'norm'




def data_read(path,data_type):
    # global data1,data2
    max_len = 0
    if data_type == data1:
        divid = 1
    elif data_type == data2:
        divid = 8
    testname = []
    for i in os.listdir(os.path.join(path+'/train',data_type)):
        l = np.array(numpy.loadtxt(os.path.join(path+'/train', data_type,i)), dtype=float)
        leng = l.shape[0]
        if leng>max_len:
            max_len=leng

    label_dict={}
    idx=0
    train_x = []

    train_y = []
    for i in os.listdir(os.path.join(path + '/train', data_type)):
        l = np.array(numpy.loadtxt(os.path.join(path + '/train', data_type, i)), dtype=float)
        l = np.nan_to_num(l)
        if not np.any(l):
            # print('gg')
            continue
        fb, _, _, lb, _ = i.split('_')

        if 'front' in fb:
            # l = numpy.loadtxt(os.path.join(path+data_type,t))
            # print(l)
            if data_type == data1:
                split_line = 1
                if l.shape[0] < max_len:
                    # print('old l ', l, l.shape)
                    l = np.pad(l, ((0, max_len - l.shape[0]), (0, 0)), 'constant', constant_values=(0, 0))
                    # print('new l ', l, l.shape)
                train_x.append(l)

                if lb not in label_dict:
                    label_dict[lb] = idx
                    idx += 1
                train_y.append(label_dict[lb])

            if data_type == data2:
                split_line = 8
                # print('la', l.shape)
                if l.shape[0]<max_len:
                    l = np.pad(l, ((0, max_len - l.shape[0]), (0, 0)), 'constant', constant_values=(0, 0))
                    # print('lb', l, l.shape)

                l = l.reshape([-1,split_line,3])

                train_x.append(l)
                if lb not in label_dict:
                    label_dict[lb] = idx
                    idx += 1
                train_y.append(label_dict[lb])


                # print('lc',l.shape)
                # aa
                # if l.shape[0]<max_len:
                #     l = np.pad(l, ((0, max_len - l.shape[0]), (0, 0)), 'constant', constant_values=(0, 0))
                #     print('lb',l,l.shape)
                #     aa
        elif 'back' in fb:

            if data_type == data1:
                split_line = 1
                # print('l b',l)
                l = l[:,[3,2,1,0]]
                # print('l f', l)
                # aa
                if l.shape[0] < max_len:
                    # print('old l ', l, l.shape)
                    l = np.pad(l, ((0, max_len - l.shape[0]), (0, 0)), 'constant', constant_values=(0, 0))
                    # print('new l ', l, l.shape)
                train_x.append(l)

                if lb not in label_dict:
                    label_dict[lb] = idx
                    idx += 1
                train_y.append(label_dict[lb])


                # print(l,nl)
            elif data_type == data2:
                split_line = 8
                nl = l.reshape([-1,split_line,3])
                # print(nl.shape)
                # aa
                split_line = 8
                # print('la', l.shape)
                if l.shape[0] < max_len:
                    l = np.pad(l, ((0, max_len - l.shape[0]), (0, 0)), 'constant', constant_values=(0, 0))
                    # print('lb', l, l.shape)

                l = l.reshape([-1, split_line, 3])

                l = l[:, [5,4,3, 2, 1, 0,7,6],:]
                # print(l.shape)


                train_x.append(l)
                if lb not in label_dict:
                    label_dict[lb] = idx
                    idx += 1
                train_y.append(label_dict[lb])

    train_x = numpy.array(train_x)
    train_y = numpy.array(train_y)
    testname = []
    test_x = []
    test_y = []
    for i in os.listdir(os.path.join(path + '/test', data_type)):
        l = np.array(numpy.loadtxt(os.path.join(path + '/test', data_type, i)), dtype=float)
        l = np.nan_to_num(l)
        if not np.any(l):
            # print('gg')
            continue
        fb, _, _, lb, _ = i.split('_')
        testname.append(i)
        if 'front' in fb:
            # l = numpy.loadtxt(os.path.join(path+data_type,t))
            # print(l)
            if data_type == data1:
                split_line = 1
                if l.shape[0] < max_len:
                    # print('old l ', l, l.shape)
                    l = np.pad(l, ((0, max_len - l.shape[0]), (0, 0)), 'constant', constant_values=(0, 0))
                    # print('new l ', l, l.shape)
                test_x.append(l)

                if lb not in label_dict:
                    label_dict[lb] = idx
                    idx += 1
                test_y.append(label_dict[lb])

            if data_type == data2:
                split_line = 8
                # print('la', l.shape)
                if l.shape[0] < max_len:
                    l = np.pad(l, ((0, max_len - l.shape[0]), (0, 0)), 'constant', constant_values=(0, 0))
                    # print('lb', l, l.shape)

                l = l.reshape([-1, split_line, 3])

                test_x.append(l)
                if lb not in label_dict:
                    label_dict[lb] = idx
                    idx += 1
                test_y.append(label_dict[lb])

                # print('lc',l.shape)
                # aa
                # if l.shape[0]<max_len:
                #     l = np.pad(l, ((0, max_len - l.shape[0]), (0, 0)), 'constant', constant_values=(0, 0))
                #     print('lb',l,l.shape)
                #     aa
        elif 'back' in fb:

            if data_type == data1:
                split_line = 1
                # print('l b',l)
                l = l[:, [3, 2, 1, 0]]
                # print('l f', l)
                # aa
                if l.shape[0] < max_len:
                    # print('old l ', l, l.shape)
                    l = np.pad(l, ((0, max_len - l.shape[0]), (0, 0)), 'constant', constant_values=(0, 0))
                    # print('new l ', l, l.shape)
                test_x.append(l)

                if lb not in label_dict:
                    label_dict[lb] = idx
                    idx += 1
                test_y.append(label_dict[lb])

                # print(l,nl)
            elif data_type == data2:
                split_line = 8
                nl = l.reshape([-1, split_line, 3])
                # print(nl.shape)
                # aa
                split_line = 8
                # print('la', l.shape)
                if l.shape[0] < max_len:
                    l = np.pad(l, ((0, max_len - l.shape[0]), (0, 0)), 'constant', constant_values=(0, 0))
                    # print('lb', l, l.shape)

                l = l.reshape([-1, split_line, 3])

                l = l[:, [5, 4, 3, 2, 1, 0, 7, 6], :]
                # print(l.shape)

                test_x.append(l)
                if lb not in label_dict:
                    label_dict[lb] = idx
                    idx += 1
                test_y.append(label_dict[lb])

    test_x = numpy.array(test_x)
    test_y = numpy.array(test_y)



    # print(train_x.shape,train_y.shape)
    # print(test_x.shape,test_y.shape)
    # print(i,data_type,label_dict)
    return train_x,train_y,test_x,test_y,label_dict,testname

class FCN_Net(nn.Module):
    """
    Based on the simpleNet above, activation function is added to the output section of each layer
    """

    def __init__(self, in_dim, out_dim):
        super(FCN_Net, self).__init__()
        self.layer1 = nn.Sequential(nn.Linear(in_dim, 300), nn.ReLU(True))
        self.layer2 = nn.Sequential(nn.Linear(300, 512), nn.ReLU(True))
        self.layer3 = nn.Sequential(nn.Linear(512, 1024), nn.ReLU(True))
        self.layer4 = nn.Sequential(nn.Linear(1024, 300), nn.ReLU(True))
        self.layer5 = nn.Sequential(nn.Linear(300, out_dim))
        """
        The function of the Sequential() function here is to combine the layers of the network together.
        """

    def forward(self, x):
        # print(x.shape)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        return x

import torch.utils.data as Data
train_x,train_y,test_x,test_y,label_dict,testname = data_read(data_path,data1)
# test_x =test_x.reshape([-1,train_x.shape[1]*train_x.shape[2]*train_x.shape[3]])
test_x =test_x.reshape([-1,test_x.shape[1]*test_x.shape[2]])
label_dict = np.load('./'+data1+'_dict.npy',allow_pickle=True).item()
nd = {v : k for k,v in label_dict.items ()}
print(label_dict,len(label_dict))


print('model test : ',data1)
print('training data shape: ' ,train_x.shape,train_y.shape)
print('test data shape: ' ,test_x.shape,test_y.shape)

test_dataset = Data.TensorDataset(torch.tensor(test_x, dtype=torch.float32), torch.tensor(test_y, dtype=torch.float32))
test_loader = Data.DataLoader(test_dataset, batch_size=1, shuffle=False)
model = FCN_Net(test_x.shape[1], len(label_dict))


# if os.path.exists("./angles_model.pkl"):
model.load_state_dict(torch.load("./angles_model.pkl"))




correct = 0.0
num = 0
ii = 0
for batch_x, batch_y in test_loader:
    # batch_x, batch_y = batch_x, batch_y.to(device)
    pred = model(batch_x)
    # loss = criterion(pred, batch_y)
    # loss_M += loss.item()
    result = F.softmax(pred,dim=1)
    # print(result[0].shape)
    lb = int(batch_y[0].item())
    print('Sample name: ',testname[ii])
    ii+=1
    for i in range(len(label_dict)):
        print('Predicted action: ',nd[i],'Predicted probability: ',result[0][i].item())
    # Find the correct rate
    pred = pred.max(1)[1]
    if pred == batch_y:
        print('Correct')
    else:
        print('Error')
    correct += (pred == batch_y).sum()
    num += len(batch_y)
acc = correct.item() / num
# loss = loss_M / len(test_loader)
print(f'model angles  acc: {acc*100:.2f}%')


train_x,train_y,test_x,test_y,label_dict,testname = data_read(data_path,data2)
# test_x =test_x.reshape([-1,train_x.shape[1]*train_x.shape[2]*train_x.shape[3]])
test_x =test_x.reshape([-1,test_x.shape[1]*test_x.shape[2]*test_x.shape[3]])
label_dict = np.load('./'+data2+'_dict.npy',allow_pickle=True).item()
nd = {v : k for k,v in label_dict.items ()}
print(label_dict,len(label_dict))


print('model test : ',data2)
print('training data shape: ' ,train_x.shape,train_y.shape)
print('test data shape: ' ,test_x.shape,test_y.shape)

test_dataset = Data.TensorDataset(torch.tensor(test_x, dtype=torch.float32), torch.tensor(test_y, dtype=torch.float32))
test_loader = Data.DataLoader(test_dataset, batch_size=1, shuffle=False)
model = FCN_Net(test_x.shape[1], len(label_dict))


# if os.path.exists("./norm_model.pkl"):
model.load_state_dict(torch.load("./norm_model.pkl"))




correct = 0.0
num = 0
ii = 0
for batch_x, batch_y in test_loader:
    # batch_x, batch_y = batch_x, batch_y.to(device)
    pred = model(batch_x)
    # loss = criterion(pred, batch_y)
    # loss_M += loss.item()
    result = F.softmax(pred,dim=1)
    # print(result[0].shape)
    lb = int(batch_y[0].item())
    print('Sample name: ',testname[ii])
    ii+=1
    for i in range(len(label_dict)):
        print('Predicted action: ',nd[i],'Predicted probability: ',result[0][i].item())
    # Find the correct rate
    pred = pred.max(1)[1]
    if pred == batch_y:
        print('Correct')
    else:
        print('Error')
    correct += (pred == batch_y).sum()
    num += len(batch_y)
acc2 = correct.item() / num
# loss = loss_M / len(test_loader)
print(f'model norm  acc: {acc2*100:.2f}%')

if acc>acc2:
    print('The angles model has the highest accuracy rate with an accuracy of',acc)
else:
    print('The norm model has the highest accuracy, with an accuracy of', acc2)
