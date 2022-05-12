import os
import shutil

data_dir = './OCR_Image'

dst_dir = './OCR_Image1'
if not os.path.exists(dst_dir):
    os.mkdir(dst_dir)

train_dir = dst_dir + '/Train_Set'
valid_dir = dst_dir + '/Valid_Set'
if not os.path.exists(train_dir):
    os.mkdir(train_dir)

if not os.path.exists(valid_dir):
    os.mkdir(valid_dir)

train_set_ratio = 0.7

for root, dirs, _ in os.walk(data_dir):
    print(root)
    for sub_dir in dirs:
        img_names = os.listdir(os.path.join(root, sub_dir))
        img_names = list(filter(lambda x: x.endswith('.png'), img_names))

        train_sub_dir = os.path.join(train_dir, sub_dir)
        if not os.path.exists(train_sub_dir):
            os.mkdir(train_sub_dir)
        # 遍历图片
        train_data_end = int(len(img_names) * train_set_ratio)
        for i in range(0, train_data_end):
            img_name = img_names[i]
            path_img = os.path.join(root, sub_dir, img_name)
            dst = os.path.join(train_sub_dir, img_name)
            shutil.copyfile(path_img, dst)

        valid_sub_dir = os.path.join(valid_dir, sub_dir)
        if not os.path.exists(valid_sub_dir):
            os.mkdir(valid_sub_dir)
        # 遍历图片
        for i in range(train_data_end, len(img_names)):
            img_name = img_names[i]
            path_img = os.path.join(root, sub_dir, img_name)
            dst = os.path.join(valid_sub_dir, img_name)
            shutil.copyfile(path_img, dst)