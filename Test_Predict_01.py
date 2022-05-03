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
import time, shutil
# from Test_Model_01 import MyData

rmb_label = {"0": 0, "1": 1, "2": 2}

print(rmb_label['2'])
############ Tain & Test
device = 'cpu'

model = ResNet50(num_classes=len(rmb_label), imgsz=128)
model = model.to(device)

last = 0.9

pth = './Resnet90.13epoch41.pt'
##### read model
if os.path.exists(pth):
    checkpoint = torch.load(pth)
    model.load_state_dict(checkpoint['model'])

############
# print(model)
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

train_dir = '/Users/edward_chen/Downloads/Test_OK/Classification/Train_Set'
valid_dir = '/Users/edward_chen/Downloads/Test_OK/Classification/Valid_Set'

test_dir = train_dir
out_dir = test_dir + '/Output'
os.makedirs(out_dir, exist_ok=True)

# train_data = MyData(data_dir=train_dir, transform=transforms.ToTensor())
# valid_data = MyData(data_dir=valid_dir, transform=transforms.ToTensor())

test_data = MyData(data_dir=test_dir, transform=transforms.ToTensor())
print(len(test_data))

model.eval()
f   = nn.Softmax(dim=1)

for i in range(len(test_data)):
    start_time = time.time()

    path_img, label = test_data.data_info[i] 
    img, _ = test_data.__getitem__(i)
    img = img[None]

    # img = img.float().to(device)
    out = model(img)    #前向算法
    out = f(out)
    conf, pred = torch.max(out,1)   #预测结果

    #print(conf, pred)
    print(i, 'time cost = ', time.time() - start_time)
    #copy issued image to another folder
    if pred != label:
        idx0 = path_img.rfind('/')
        img_name = path_img[idx0+1:]
        img_name = str(conf.data.numpy()[0]) +'_' +str(pred[0])+'_'+str(label)+ img_name

        shutil.copyfile(path_img, out_dir+'/'+img_name)
        #print(img_name)


