import torch
import torch.nn as nn
import time, shutil, os, cv2
from torchvision import transforms
from ClassicNetwork.ResNet import ResNet50
from PIL import Image
import numpy as np

#### Get CAM results
# def hook_feature(module, input, output):
#     features_blobs.append(output.data.cpu().numpy())

def returnCAM(feature_conv, weight_softmax, class_idx):
    bz, nc, h, w = feature_conv.shape        #1,2048,7,7
    output_cam = []
    for idx in class_idx:  #只输出预测概率最大值结果不需要for循环
        feature_conv = feature_conv.reshape((nc, h*w))
        cam = weight_softmax[idx].dot(feature_conv.reshape((nc, h*w)))  #(2048, ) * (2048, 7*7) -> (7*7, ) （n,）是一个数组，既不是行向量也不是列向量
        cam = cam.reshape(h, w)
        cam_img = (cam - cam.min()) / (cam.max() - cam.min())  #Normalize
        cam_img = np.uint8(255 * cam_img)                      #Format as CV_8UC1 (as applyColorMap required)

        #output_cam.append(cv2.resize(cam_img, size_upsample))  # Resize as image size
        output_cam.append(cam_img)
    return output_cam

File_path = '/Users/edward_chen/Downloads/Test_OK/Cls/Valid_Set/1'
# File_name = '_A8D48549-C49C-4284-9091-C5F60CB75058_Region1_r0_c0.jpg'

CAM_RESULT_PATH = os.path.join(File_path, 'CAMs')   #CAM结果的存储地址
os.makedirs(CAM_RESULT_PATH, exist_ok=True)
os.makedirs(CAM_RESULT_PATH +'/0', exist_ok=True)
os.makedirs(CAM_RESULT_PATH +'/1', exist_ok=True)

label  = {"0": 0, "1": 1}
class_ = ['Screw', 'No_Screw']

############ Tain & Test
device = 'cpu'

model = ResNet50(num_classes=len(label), imgsz=256)
model = model.to(device)

#### load model
pth        = './Resnet99.83epoch8.pt'
checkpoint = torch.load(pth)
model.load_state_dict(checkpoint['model'])

#### load model to last feature map to create CAM feature (Classification Activation Map)
model_features = nn.Sequential(*list(model.children())[:-2])

model.eval()
model_features.eval()

trans = transforms.Compose([transforms.ToTensor(),])
f1    = nn.Softmax(dim=1)

for File_name in os.listdir(File_path):
    if File_name.endswith('.jpg'):

        ## open image and transform to tensor
        imgt = Image.open(os.path.join(File_path, File_name))
        img  = trans(imgt)
        img  = img[None]

        img = img.float().to(device)

        ## model predict
        start_time = time.time()
        out = model(img)    #前向算法
        out = f1(out)
        conf, pred = torch.max(out,1)   #预测结果

        end_time = time.time()

        # print(conf, pred)
        print('time cost = ', end_time - start_time)

        ### get feature map
        features   = model_features(img).detach().cpu().numpy() 
        fc_weights = model.state_dict()['fc.weight'].cpu().numpy()

        # print(features.shape)
        # print(fc_weights.shape)

        CAMs = returnCAM(features, fc_weights, pred)  #输出预测概率最大的特征图集对应的CAM
        # print(img_name + ' output for the top1 prediction: %s' % class_[idx[0]])

        #img  = cv2.imread(os.path.join(File_path, File_name))
        imgt = cv2.cvtColor(np.asarray(imgt),cv2.COLOR_RGB2BGR) 
        height, width, _ = imgt.shape  #读取输入图片的尺寸
        heatmap = cv2.applyColorMap(cv2.resize(CAMs[0], (width, height)), cv2.COLORMAP_JET)  #CAM resize match input image size
        result = heatmap * 0.3 + imgt * 0.5    #比例可以自己调节

        text = '%s %.2f%%' % (class_[pred], conf[0]*100) 				 #激活图结果上的文字显示
        cv2.putText(result, text, (20, height-10), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.9,
                    color=(123, 222, 238), thickness=2, lineType=cv2.LINE_AA)

        image_name_ = File_name[:-4]

        if pred == 0:
            cv2.imwrite(CAM_RESULT_PATH + '/0/' + image_name_ + '_pred_' + class_[pred] + '.jpg', result)  #写入存储磁盘
        else:
            cv2.imwrite(CAM_RESULT_PATH + '/1/' + image_name_ + '_pred_' + class_[pred] + '.jpg', result)  #写入存储磁盘
