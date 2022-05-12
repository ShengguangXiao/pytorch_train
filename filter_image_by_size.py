import os

data_dir = './OCR_Image'
for root, dirs, _ in os.walk(data_dir):
    #print(root)
    for sub_dir in dirs:
        #print(sub_dir)
        img_names = os.listdir(os.path.join(root, sub_dir))
        img_names = list(filter(lambda x: x.endswith('.png'), img_names))
                # 遍历图片
        for i in range(len(img_names)):
            img_name = img_names[i]
            path_img = os.path.join(root, sub_dir, img_name)
            fileSize = os.path.getsize(path_img)
            #print(f"File {path_img} size {fileSize}")
            if (fileSize < 1000):
                os.remove(path_img)
                print(f"Removed file: {path_img}")