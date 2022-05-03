import math, os
import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy
from PIL import Image
from scipy import ndimage
import torch
import torch.nn as nn
from cnn_utils import *
from torch import nn,optim
from torch.utils.data import DataLoader,Dataset
from torchvision import transforms
from ClassicNetwork.ResNet import ResNet50

#matplotlib inline
np.random.seed(1)
torch.manual_seed(1)
batch_size = 64
learning_rate = 0.009
num_epocher = 100
pre_epoch = 0

# X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()
# X_train = X_train_orig/255.
# X_test = X_test_orig/255.

# class MyData(Dataset): #继承Dataset
#     def __init__(self, data, y, transform=None): #__init__是初始化该类的一些基础参数
#         self.transform = transform #变换
#         self.data = data
#         self.y = y
    
#     def __len__(self):#返回整个数据集的大小
#         return len(self.data)
    
#     def __getitem__(self,index):#根据索引index返回dataset[index]
#         sample = self.data[index]
#         if self.transform:
#             sample = self.transform(sample)#对样本进行变换
#         return sample, self.y[index] #返回该样本
    
# train_dataset = MyData(X_train, Y_train_orig[0],
#     transform=transforms.ToTensor())
# test_dataset = MyData(X_test, Y_test_orig[0],
#     transform=transforms.ToTensor())
# train_loader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
# test_loader = DataLoader(test_dataset,batch_size=batch_size,shuffle=False)
rmb_label = {"0": 0, "1": 1}

class MyData(Dataset):
    @staticmethod
    def get_img_info(data_dir):
        data_info = list()
        # data_dir 是训练集、验证集或者测试集的路径
        for root, dirs, _ in os.walk(data_dir):
            # 遍历类别
            # dirs ['1', '100']
            for sub_dir in dirs:
                # 文件列表
                img_names = os.listdir(os.path.join(root, sub_dir))
                # 取出 jpg 结尾的文件
                img_names = list(filter(lambda x: x.endswith('.jpg'), img_names))
                # 遍历图片
                for i in range(len(img_names)):
                    img_name = img_names[i]
                    # 图片的绝对路径
                    path_img = os.path.join(root, sub_dir, img_name)
                    # 标签，这里需要映射为 0、1 两个类别
                    label = rmb_label[sub_dir]
                    # 保存在 data_info 变量中
                    data_info.append((path_img, int(label)))
        return data_info

    def __init__(self, data_dir, transform=None):
        """
        rmb面额分类任务的Dataset
        :param data_dir: str, 数据集所在路径
        :param transform: torch.transform，数据预处理
        """
        # data_info存储所有图片路径和标签，在DataLoader中通过index读取样本
        self.data_info = self.get_img_info(data_dir)
        self.transform = transform
    
    def __len__(self):
        return len(self.data_info)

    def __getitem__(self, index):
        # 通过 index 读取样本
        path_img, label = self.data_info[index]
        # 注意这里需要 convert('RGB')
        img = Image.open(path_img).convert('RGB')     # 0~255
        if self.transform is not None:
            img = self.transform(img)   # 在这里做transform，转为tensor等等
        # 返回是样本和标签
        return img, label

train_dir = '/Users/edward_chen/Downloads/Test_OK/Cls/Train_Set'
valid_dir = '/Users/edward_chen/Downloads/Test_OK/Cls/Valid_Set'

# train_data = MyData(data_dir=train_dir, transform=train_transform)
# valid_data = MyData(data_dir=valid_dir, transform=valid_transform)

train_data = MyData(data_dir=train_dir, transform=transforms.ToTensor())
valid_data = MyData(data_dir=valid_dir, transform=transforms.ToTensor())

# img1, label1 = train_data.__getitem__(300)

# print(len(train_data))
# print(len(valid_data))

# 构建DataLoder
# 其中训练集设置 shuffle=True，表示每个 Epoch 都打乱样本
train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(dataset=valid_data, batch_size=batch_size, shuffle=False)

############ Tain & Test
device = 'cpu'

def test():
    model.eval()    #需要说明是否模型测试
    eval_loss = 0
    eval_acc = 0
    for data in valid_loader:
        img,label = data
        img = img.float().to(device)
        label = label.long().to(device)
        out = model(img)    #前向算法
        loss = criterion(out,label) #计算loss
        eval_loss += loss.item() * label.size(0)    #total loss
        _,pred = torch.max(out,1)   #预测结果
        num_correct = (pred == label).sum() #正确结果
        eval_acc += num_correct.item()  #正确结果总数

    print('Test Loss:{:.6f},Acc: {:.6f}'
          .format(eval_loss/ (len(valid_data)),eval_acc * 1.0/(len(valid_data))))

    acc = eval_acc * 1.0/(len(valid_data))
    return acc

##### import model
# model = ResModel(6)
model = ResNet50(num_classes=len(rmb_label), imgsz = 256)
model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.8)

last = 0.9

pth = './Resnet91.44epoch42.pt'
##### read model
# if os.path.exists(pth):
#     checkpoint = torch.load(pth)
#     model.load_state_dict(checkpoint['model'])
#     optimizer.load_state_dict(checkpoint['optimizer'])
#     pre_epoch = checkpoint['epoch']
    #last = ptt / 100.0

#开始训练
for epoch in range(pre_epoch, num_epocher):
    model.train()
    running_loss = 0.0
    running_acc = 0.0
    for i,data in enumerate(train_loader,1):
        img,label = data
        img = img.float().to(device)
        label = label.long().to(device)
        #前向传播
        out = model(img)
        loss = criterion(out,label) #loss
        running_loss += loss.item() * label.size(0)
        _,pred = torch.max(out,1)   #预测结果
        num_correct = (pred == label).sum() #正确结果的数量
        running_acc += num_correct.item()   #正确结果的总数
        
        optimizer.zero_grad()   #梯度清零
        loss.backward() #后向传播计算梯度
        optimizer.step()    #利用梯度更新W，b参数
    #打印一个循环后，训练集合上的loss和正确率
    if (epoch+1) % 1 == 0:
        print('Train{} epoch, Loss: {:.6f},Acc: {:.6f}'.format(epoch+1,running_loss / (len(train_data)),
                                                               running_acc / (len(train_data))))
        now = test()

    ## save model
    # if epoch == 0:
    #     state = {'model':model.state_dict(), 'optimizer':optimizer.state_dict(), 'epoch':epoch}
    #     torch.save(state, './Resnet'+str(int(now*10000)/100)+'epoch'+str(epoch)+'.pt')
    #     last = now
    
    if now > last:
        state = {'model':model.state_dict(), 'optimizer':optimizer.state_dict(), 'epoch':epoch}
        torch.save(state, './Resnet'+str(int(now*10000)/100)+'epoch'+str(epoch)+'.pt')
        last = now



    