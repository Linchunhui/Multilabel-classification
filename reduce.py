import pandas as pd
import numpy as np
import os
from keras.preprocessing import image
import tensorflow as tf
import datetime
import cv2
#from MobilenetV3 import mobilenet_v3_small
from net.MobileNetV2 import MobileNetV2
from net.MobileNetV1 import MobileNetV1
import shutil

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # gpu编号
config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # 设置最小gpu使用量

from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
try:
    import urllib2 as urllib
except ImportError:
    import urllib.request as urllib

names = {0: 'cat', 1: 'dog', 2: 'other'}
IMG_W = 224

def std(image):
    mean=np.zeros([224,224,3])
    std=np.zeros([224,224,3])
    mean,std=cv2.meanStdDev(image,mean,std)

def init_tf(logs_train_dir):
    global sess, pred, x
    # process image
    x = tf.placeholder(tf.float32, shape=[IMG_W, IMG_W, 3])
    x_norm = tf.image.per_image_standardization(x)
    x_4d = tf.reshape(x_norm, [-1, IMG_W, IMG_W, 3])
    model = MobileNetV1(x_4d, 3, is_training=False)
    logit = model.output
    print("logit", np.shape(logit))
    pred = tf.nn.sigmoid(logit)

    saver = tf.train.Saver()
    sess = tf.Session()
    saver.restore(sess, logs_train_dir)
    print('load model done...')


def evaluate_image2(img_path):
    img = image.load_img(img_path, target_size=(IMG_W, IMG_W))
    image_array = image.img_to_array(img)
    prediction = sess.run(pred, feed_dict={x: image_array})
    return prediction

def func(a):
    for i in range(len(a)):
        a[i] = int(a[i])
    return a

def output1(pred, threshold=0.4):
    pred = pred[0]
    flag = [i >= threshold for i in pred]
    a = func(flag)
    return a

def output(pred, threshold=0.4):
    cate = ['cat', 'dog', 'other']
    pred = pred[0]
    flag1 = [i >= threshold for i in pred]
    res1 = []
    for i in range(len(flag1)):
        if flag1[i] == True:
            res1.append(cate[i])
    return res1

def output_argmax(a):
    return np.argmax(a)

def read_file_all(data_dir_path):
    file_list = []
    res_list = []
    res_cate_list = []
    pred = []
    pred1 = []
    result = pd.DataFrame()
    count = 0
    count1 = 0
    count2 = 0
    count3 = 0
    i = 0
    for f in os.listdir(data_dir_path):
        if f == 'Thumbs.db':
            continue
        if f.split(".")[-1]!="jpg":
            continue
        image_path = os.path.join(data_dir_path, f)
        # print(i,image_path)
        i += 1
        dog_error = "D:/test_error/dog"
        cat_error = "D:/test_error/cat"
        if os.path.isfile(image_path):
            preds = evaluate_image2(image_path)
            res_int = output1(preds)  # 数字
            #if res_int[1]!=1 or res_int[0]==1:
             #   shutil.copy(image_path, os.path.join(dog_error, f))
            #if res_int[0]!=1 or res_int[1]==1:
                #shutil.copy(image_path, os.path.join(cat_error, f))
            res_cate = output(preds)  # 类别
            print("img:{} , pred:{} , pred_int:{}, preds:{},{}/{}" .format(image_path, res_cate, res_int,preds,i,
                                                                  len(os.listdir(data_dir_path))))
            # print(res_int)
            file_list.append(image_path)
            res_list.append(res_int)
            res_cate_list.append(res_cate)
            pred.append(preds)
            pred1.append(preds[0][1])
            # count=0
            if res_cate == ['cat', 'dog']:
                count += 1
            elif res_cate == ['cat'] or res_cate==['cat', 'other']:
                count1 += 1
            elif res_cate == ['dog'] or res_cate==['dog', 'other']:
                count2 += 1
            else:
                count3 += 1
    result['filename'] = file_list
    result['res_int'] = res_list
    result['res_cate'] = res_cate_list
    result['pred'] = pred
    result['pred1'] = pred1
    result.to_csv(r"D:\project\ShuffleNet\csv\train_complex\dog11.csv", index=False, encoding='utf-8')
    print('cat cout:{} dog count:{},cat_dog count:{} other:{}/all count:{}'.format(count1, count2, count, count3,
                                                                                   len(res_list)))
    print("accuracy:{}".format((count1 / (len(res_list)))))
    print("accuracy:{}".format((count2/(len(res_list)))))


if __name__ == "__main__":
    logs_train_dir = 'model_save/mobilenetv1/train2/model.ckpt-250000'
    init_tf(logs_train_dir)
    #data_dir = 'D:/RoadMapSample/'
    #data_dir = 'D:/data/cat_dog/'
    #data_dir = 'D:/val/Dog/'
    #data_dir = "D:/cat/cat1"
    data_dir = "D:/train/Dog"
    read_file_all(data_dir)