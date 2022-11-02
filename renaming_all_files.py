import math, os
import sys

def rename_one_folder(parent_path, folder_name):
    folder_path = parent_path + folder_name + '/'
    print(folder_path)
    img_names = os.listdir(folder_path)
    img_names = list(filter(lambda x: x.endswith('.png'), img_names))

    for i in range(len(img_names)):
        new_img_name = folder_name + "-index-" + str(i + 1) + ".png"
        os.rename(folder_path + img_names[i], folder_path + new_img_name)


if __name__ == '__main__':
    try:
        [_, parent_path] = sys.argv
    except:
        print("Please input the parent path")
        exit

    if not parent_path.endswith('/'):
        parent_path = parent_path + '/'

    folder_paths = [ f.path for f in os.scandir(parent_path) if f.is_dir() ]

    for folder_path in folder_paths:
        index = folder_path.rfind('/')
        folder_name = folder_path[index + 1:len(folder_path)]

        rename_one_folder(parent_path, folder_name)