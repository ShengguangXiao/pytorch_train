import os, cv2, math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torchvision import transforms
import time
from ClassicNetwork.ResNet import ResNet50

File_path = '/Users/edward_chen/Library/Mobile Documents/com~apple~CloudDocs/Desktop/QSMC/ConnectorAOI_CosmeticInspection/Iphone_AOI/Station1/SplitRegion/FOV2_Region2/NG'
File_List = os.listdir(File_path)
File_List = list(filter(lambda x: x.endswith('.jpg') == True, File_List))

Out_path = File_path + '/Output'
os.makedirs(Out_path, exist_ok=True)

imgsz    = (256, 256)
min_ovlp = 0.07

def GetSplitBlock(img, imgsz):
    m, n = img.shape[:2]

    ms, ns = math.ceil(m/imgsz[0]), math.ceil(n/imgsz[1])

    if ms < m/imgsz[0] + (ms-1)*min_ovlp: ms += 1
    if ns < n/imgsz[1] + (ns-1)*min_ovlp: ns += 1

    ovlp_row = 1 if ms == 1 else (imgsz[0]*ms - m)/(ms-1)
    ovlp_col = 1 if ns == 1 else (imgsz[1]*ns - n)/(ns-1)

    row = np.arange(0, m, imgsz[0]-ovlp_row)
    col = np.arange(0, n, imgsz[1]-ovlp_col)

    row = np.array(row[:-1], dtype=np.int16)
    col = np.array(col[:-1], dtype=np.int16)

    row[-1] = m - imgsz[0]
    col[-1] = n - imgsz[1]

    return m, n, row, col

## load model
label  = {"0": 0, "1": 1}
class_ = ['Screw', 'No_Screw']

############ Tain & Test
device = 'cpu'

model = ResNet50(num_classes=len(label), imgsz=256)
model = model.to(device)

#### load model
pth        = './Resnet99.72epoch73.pt'
checkpoint = torch.load(pth)
model.load_state_dict(checkpoint['model'])
model.eval()

### split image, convert to pytorch format, do prediction
trans = transforms.Compose([transforms.ToTensor(),])
f1    = nn.Softmax(dim=1)

# print(File_List[0])
########## split image
for file in File_List:
    imgt = plt.imread(os.path.join(File_path, file))
    m, n, row, col = GetSplitBlock(imgt, imgsz)

    img = torch.tensor([])
    for i in range(len(row)):
        for j in range(len(col)):
            # fname_ij = fname + '_r' + str(i) + '_c'+str(j)
            img_ij = imgt[row[i]:row[i]+imgsz[0], col[j]:col[j]+imgsz[1], :]

            ######## create to tensor
            img0t = trans(img_ij)
            img0t = img0t[None]

            img = torch.cat([img, img0t])  ## concat to [8, 3, 256, 256]

    # model predict
    start_time = time.time()
    out = model(img)    #前向算法
    out = f1(out)
    conf, pred = torch.max(out,1)   #预测结果

    end_time = time.time()
    print('time cost = ', end_time - start_time)

    imgt = cv2.cvtColor(np.asarray(imgt),cv2.COLOR_RGB2BGR)
    for i, p in enumerate(pred):
        if p == 0: ## draw bbox
            imgt = cv2.rectangle( imgt,
                (col[i], row[0]), (col[i]+imgsz[1], row[0]+imgsz[0]),
                [0, 0, 255],
                6, )

    cv2.imwrite(Out_path+'/'+file, imgt)
    

