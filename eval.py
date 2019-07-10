import pandas as pd
import numpy as np
import os
from keras.preprocessing import image
import tensorflow as tf
#from MobilenetV3 import mobilenet_v3_small
from net.net import ShuffleNetV2

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

def init_tf(logs_train_dir):#='model_save/train1/model.ckpt-120000'):
    global sess, pred, x
    # process image
    x = tf.placeholder(tf.float32, shape=[IMG_W, IMG_W, 3])
    #x = tf.image.random_flip_left_right(x)
    #x = tf.image.random_flip_up_down(x)
    #x = tf.image.random_brightness(x, max_delta=32 / 255.0)
    #x_norm= tf.image.random_hue(x, max_delta=0.05)
    x_norm = tf.image.per_image_standardization(x)

    x_4d = tf.reshape(x_norm, [-1, IMG_W, IMG_W, 3])
    # predict
    #logit,_ =mobilenet_v3_small(x_4d,classes_num=3,multiplier=1.0,is_training=False,reuse=None)
    model = ShuffleNetV2(x_4d, 3, model_scale=2.0, is_training=False)
    logit = model.output
    print("logit", np.shape(logit))
    pred = tf.nn.sigmoid(logit)

    saver = tf.train.Saver()
    sess = tf.Session()
    saver.restore(sess, logs_train_dir)
    print('load model done...')


def evaluate_image(img_path):
    img = image.load_img(img_path, target_size=(IMG_W, IMG_W))
    image_array = image.img_to_array(img)
    prediction = sess.run(pred, feed_dict={x: image_array})
    return prediction

def func(a):
    for i in range(len(a)):
        # a[i]=1 if a[i]==True else 0
        a[i] = int(a[i])
    return a
def func1(a):
    if a[0]==1 and a[1]==0:
        return 0
    if a[0]==0 and a[1]==1:
        return 1
    if a[0]==1 and a[1]==1:
        return 3
    else:
        return 2
def output(pred, threshold=0.4):
    pred = pred[0]
    flag = [i >= threshold for i in pred]
    a = func(flag)
    b=func1(a)
    return a,b

def output1(pred, threshold=0.4):
    cate = ['cat', 'dog', 'other']
    pred = pred[0]
    flag1 = [i >= threshold for i in pred]
    res1 = []
    for i in range(len(flag1)):
        if flag1[i] == True:
            res1.append(cate[i])
    return res1

'''def read_file_all1(ImgList,LabelList):
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
        if os.path.isfile(image_path):
            preds = evaluate_image2(image_path)
            res_int = output1(preds)  # 数字
            res_cate = output(preds)  # 类别
            print(i, image_path, preds, res_cate, res_int)
            # print(res_int)
            file_list.append(image_path)
            res_list.append(res_int)
            res_cate_list.append(res_cate)
            # count=0
            if res_cate == ['cat', 'dog']:
                count += 1
            elif res_cate == ['cat']:
                count1 += 1
            elif res_cate == ['dog']:
                count2 += 1
            else:
                count3 += 1
    result['filename'] = file_list
    result['res_int'] = res_list
    result['res_cate'] = res_cate_list
    #result.to_csv("D:/res/result3.csv", index=False, encoding='utf-8')
    print('cat cout:{} dog count:{},cat_dog count:{} other:{}/all count:{}'.format(count1, count2, count, count3,
                                                                                   len(res_list)))'''
def read_file_all(ImgList,LabelList):
    CountCat0 = 0
    CountCat1 = 0
    CountDog0 = 0
    CountDog1 = 0
    CountOther0 = 0
    CountOther1 = 0
    for i in range(len(ImgList)):
        prediction=evaluate_image(ImgList[i])
        pred1, pred = output(prediction)
        label = int(LabelList[i])
        print("img:{} ,prediction:{},pred1:{},pred:{} ,label:{}, {}/{}".format(ImgList[i], prediction,pred1, pred, label, i, len(ImgList)))
        if label == pred:
            if label == 0:
                CountCat0 += 1
                CountCat1 += 1
            if label == 1:
                CountDog0 += 1
                CountDog1 += 1
            if label == 2:
                CountOther0 += 1
                CountOther1 += 1
        else:
            if label==0:
                CountCat1+=1
            if label==1:
                CountDog1+=1
            if label==2:
                CountOther1+=1
    AccCat = (CountCat0/CountCat1)*100.0
    AccDog = (CountDog0/CountDog1)*100.0
    AccOther = (CountOther0/CountOther1)*100.0
    AccAll = ((CountOther0+CountDog0+CountCat0)/(CountOther1+CountDog1+CountCat1))*100.0
    print("accuracy cat:{} dog:{} other:{} all:{}".format(AccCat, AccDog, AccOther, AccAll))


if __name__ == "__main__":
    log_dir = 'model_save/train9/model.ckpt-254774'
    init_tf(log_dir)
    val=pd.read_csv("D:/project/ShuffleNet/csv/val6.csv")
    imglist=val['name']
    labellist=val['label']
    read_file_all(imglist,labellist)