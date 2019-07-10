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
    #x = tf.image.random_flip_left_right(x)
    #x = tf.image.random_flip_up_down(x)
    #x = tf.image.random_brightness(x, max_delta=32 / 255.0)
    #x = tf.image.random_contrast(x, lower=0.5, upper=1.5)
    #x = tf.image.random_hue(x, max_delta=0.05)
    #x = tf.image.random_saturation(x, lower=0.5, upper=1.5)
    x_norm = tf.image.per_image_standardization(x)
    #x_norm=tf.identity(x)
    x_4d = tf.reshape(x_norm, [-1, IMG_W, IMG_W, 3])
    # predict
    #logit,_ =mobilenet_v3_small(x_4d,classes_num=3,multiplier=1.0,is_training=False,reuse=None)
    #model = ShuffleNetV2(x_4d, 3, model_scale=2.0, is_training=False)
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
        # a[i]=1 if a[i]==True else 0
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
    result = pd.DataFrame()
    count = 0
    count1 = 0
    count2 = 0
    count3 = 0
    i = 0
    for f in os.listdir(data_dir_path):
        if f == 'Thumbs.db':
            pass
        image_path = os.path.join(data_dir_path, f)
        # print(i,image_path)
        i += 1
        dog_error = "D:/test_error/dog1"
        cat_error = "D:/test_error/cat1"
        cat_dog_error = "D:/test_error/cat_dog"
        if os.path.isfile(image_path):
            preds = evaluate_image2(image_path)
            res_int = output1(preds)  # 数字
            #if res_int[1]!=1 or res_int[0]==1:
               #shutil.copy(image_path, os.path.join(dog_error, f))
            if res_int[0]!=1 or res_int[1]==1:
                shutil.copy(image_path, os.path.join(cat_error, f))
            #if res_int[0]!=1 or res_int[1]!=1:
                #shutil.copy(image_path, os.path.join(cat_dog_error, f))
            res_cate = output(preds)  # 类别
            print("img:{} , pred:{} , pred_int:{}, preds:{},{}/{}" .format(image_path, res_cate, res_int,preds,i,
                                                                  len(os.listdir(data_dir_path))))
            # print(res_int)
            file_list.append(image_path)
            res_list.append(res_int)
            res_cate_list.append(res_cate)
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
    result.to_csv("D:/res/res1.9.csv", index=False, encoding='utf-8')
    print('cat cout:{} dog count:{},cat_dog count:{} other:{}/all count:{}'.format(count1, count2, count, count3,
                                                                                   len(res_list)))
    print("accuracy:{}".format((count1 / (len(res_list)))))
    print("accuracy:{}".format((count2/  (len(res_list)))))

def read_file_all1(data_dir):
    file_list = []
    res_list = []
    res_cate_list = []
    res_index_list=[]
    flag=[]
    result = pd.DataFrame()
    count = 0
    count1 = 0
    count2 = 0
    count3 = 0
    #i = 0
    data=pd.read_csv(data_dir)
    #data=data[]
    data_list=data['name']
    label_list=data['label']
    for i in range(len(data_list)):
        #if f == 'Thumbs.db':
           # pass
        image_path = data_list[i]
        # print(i,image_path)
        #i += 1
        if os.path.isfile(image_path):
            preds = evaluate_image2(image_path)
            res_int = output1(preds)  # 数字
            res_cate = output(preds)  # 类别
            res_index = output_argmax(preds)
            time=datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print("time:{},img:{} , pred:{} , pred_int:{}, preds:{},pred_index:{},label:{},{}/{}" .format(time,image_path, res_cate, res_int,preds,
                                                                                                  res_index,label_list[i],i,len(data_list)))
            # print(res_int)
            file_list.append(image_path)
            res_list.append(res_int)
            res_cate_list.append(res_cate)
            res_index_list.append(res_index)
            flag.append(int(res_index)==int(label_list[i]))
            # count=0
            if res_cate == ['cat', 'dog']:
                count += 1
            elif res_cate == ['cat'] or res_cate==['cat', 'other']:
                count1 += 1
            elif res_cate == ['dog'] or res_cate==['dog', 'other']:
                count2 += 1
            else:
                count3 += 1
    result['name'] = file_list
    result['res_int'] = res_list
    result['res_cate'] = res_cate_list
    result['res_index'] = res_index_list
    result['label'] = label_list
    result['flag'] = flag
    #result.to_csv("D:/project/ShuffleNet/csv/mobile_result2.csv", index=False, encoding='utf-8')
    print('cat cout:{} dog count:{},cat_dog count:{} other:{}/all count:{}'.format(count1, count2, count, count3,
                                                                                   len(res_list)))
    print("accuracy:{}".format((count1 / (len(res_list)))))
    print("accuracy:{}".format((count2/(len(res_list)))))

if __name__ == "__main__":
    logs_train_dir = 'model_save/mobilenetv1/train4/model.ckpt-140000'
    init_tf(logs_train_dir)
    #data_dir = 'D:/RoadMapSample/'
    data_dir = 'D:/data/dog2/'
    #data_dir = 'D:/val/Dog/'
    #data_dir = "D:/cat/"
    read_file_all(data_dir)