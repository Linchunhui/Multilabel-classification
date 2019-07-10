import os
import numpy as np
import tensorflow as tf
import glob
import sys
import model
from nets.mobilenet import mobilenet_v2
import tensorflow.contrib.slim as slim
sys.path.append("project/ShuffleNet")
import pandas as pd
'''label_dict, label_dict_res = {}, {}
# 手动指定一个从类别到label的映射关系
with open("label.txt", 'r') as f:
    for line in f.readlines():
        folder, label = line.strip().split(':')[0], line.strip().split(':')[1]
        label_dict[folder] = label
        label_dict_res[label] = folder
print(label_dict)'''

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

def get_files1(file_dir):
    image_list, label_list = [], []
    for f in os.listdir(file_dir):
        #for img in glob.glob(os.path.join(file_dir, label, "*.jpg")):
        image_list.append(os.path.join(file_dir,f))
            #label_list.append(int(label_dict[label]))
    print('There are %d data' % (len(image_list)))
    #temp = np.array([image_list])
    #temp = temp.transpose()
    #np.random.shuffle(temp)
    #image_list = list(temp[:, 0])
    #label_list = list(temp[:, 1])
    #label_list = [int(i) for i in label_list]
    return image_list
def a():
    train_dir = "D:/train"
    a,b=get_files(train_dir)
    img_list=pd.DataFrame()
    img_list['name']=a
    img_list['label']=b
    img_list.to_csv("D:/res/img2.csv",index=False,encoding='utf-8')
    print(1)

def a1():
    #train_dir = r"D:\project\ShuffleNet\spy\catdog5"
    train_dir = r"D:\dataset\cat_dog\cat_dog"
    a = get_files1(train_dir)
    return a

if __name__=="__main__":
    #a()

    #img_list=pd.read_csv("D:/res/img1.csv")
    #img_list=pd.read_csv("D:/project/ShuffleNet/csv/img1.csv")
    #a=img_list['name']
    a=a1()
    print(len(a))
    print(1)
    filename = tf.placeholder(tf.string, [], name='filename')
    image_file = tf.read_file(filename)
    # Decode the image as a JPEG file, this will turn it into a Tensor
    image = tf.image.decode_jpeg(image_file)  # 图像解码成矩阵
    sess=tf.Session()
    for i in range(len(a)):
        print(a[i], '{}/{}'.format(i,len(a)))
        img=sess.run(image, feed_dict={filename:a[i]})

