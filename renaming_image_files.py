import math, os
import sys

if __name__ == '__main__':
    try:
        [_, folder_path, start_index] = sys.argv
    except:
        print("Please input folder and start index")
        exit

    img_names = os.listdir(folder_path)
    img_names = list(filter(lambda x: x.endswith('.png'), img_names))

    index = int(start_index)
    for i in range(len(img_names)):
        img_name = img_names[i]
        new_img_name = img_name[0] + "-index-" + str(index) + ".png"
        os.rename(folder_path + img_name, folder_path + new_img_name)
        index = index + 1
