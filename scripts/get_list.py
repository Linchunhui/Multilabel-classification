import pandas as pd
import os
import glob
import numpy as np

def get_files(file_dir):
    image_list, label_list = [], []
    for label in os.listdir(file_dir):
        for img in glob.glob(os.path.join(file_dir, label, "*.jpg")):
            image_list.append(img)
            label_list.append(label_dict[label])
    print('There are %d data' % (len(image_list)))
    temp = np.array([image_list, label_list])
    temp = temp.transpose()
    np.random.shuffle(temp)
    image_list = list(temp[:, 0])
    label_list = list(temp[:, 1])
    label_list = [i for i in label_list]
    return image_list, label_list


label_dict, label_dict_res = {}, {}
# 手动指定一个从类别到label的映射关系
with open("D:/project/ShuffleNet/label.txt", 'r') as f:
    for line in f.readlines():
        folder, label = line.strip().split(':')[0], line.strip().split(':')[1]
        label_dict[folder] = label
        label_dict_res[label] = folder
print(label_dict)

if __name__=="__main__":
    data_dir = "D:/train3"
    img,label = get_files(data_dir)
    list1 = pd.DataFrame()
    list1['name'] = img
    list1['label'] = label
    list1.to_csv("D:/project/ShuffleNet/csv/img1.2.csv", index=False,encoding="utf-8")