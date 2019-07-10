import pandas as pd
from PIL import Image
import numpy as np
import os,glob
label_dict, label_dict_res = {}, {}
import warnings
warnings.filterwarnings("error", category=UserWarning)
import pandas as pd
import piexif
#piexif.remove(img)
'''
# 手动指定一个从类别到label的映射关系
with open("label.txt", 'r') as f:
    for line in f.readlines():
        folder, label = line.strip().split(':')[0], line.strip().split(':')[1]
        label_dict[folder] = label
        label_dict_res[label] = folder
print(label_dict)

def get_files(file_dir):
    image_list, label_list = [], []
    for label in os.listdir(file_dir):
        for img in glob.glob(os.path.join(file_dir, label, "*.jpg")):
            image_list.append(img)
            label_list.append(int(label_dict[label]))
    print('There are %d data' % (len(image_list)))
    temp = np.array([image_list, label_list])
    temp = temp.transpose()
    np.random.shuffle(temp)
    image_list = list(temp[:, 0])
    label_list = list(temp[:, 1])
    label_list = [int(i) for i in label_list]
    return image_list, label_list
train_dir = "D:/train"

train, train_label = get_files(train_dir)
x=pd.DataFrame()
x['filename']=train
x['label']=train_label'''

def corrupt():
    x=pd.read_csv('D:/res/img_list.csv')
    print(1)
    #img_list = pd.read_csv("D:/res/img.csv")
    a = x['filename']
    print(a[:10])
    for i in range(len(a)):
        path=a[i]
        print('{},{}/{}'.format(i,path, len(a)))
        try:
            img = Image.open(path)
        except IOError:
            print(path)
        try:
            #img = Image.open(path)
            img = np.array(img)
        except:
            print('corrupt img', path)
        #print('{}/{}'.format(i,len(a)))

def get_files(file_dir):
    image_list, label_list = [], []
    for label in os.listdir(file_dir):
        for img in glob.glob(os.path.join(file_dir, label, "*.jpg")):
            image_list.append(img)
            label_list.append(int(label_dict[label]))
    print('There are %d data' % (len(image_list)))
    temp = np.array([image_list, label_list])
    temp = temp.transpose()
    np.random.shuffle(temp)
    image_list = list(temp[:, 0])
    label_list = list(temp[:, 1])
    label_list = [int(i) for i in label_list]
    return image_list, label_list

train_dir = "D:/train/Other"

#train, train_label = get_files(train_dir)

if __name__=="__main__":
    train, train_label = get_files(train_dir)
    img_list=train[:20000]
    for i in range(10):
        print(img_list[i])