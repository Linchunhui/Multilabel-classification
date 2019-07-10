# coding:utf-8
import os, cv2
import numpy as np
import tensorflow as tf
import glob

from tensorflow.python.platform import gfile

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # use gpu 0

label_dict, label_dict_res = {}, {}
with open("label1.txt", 'r') as f:
    for line in f.readlines():
        folder, label = line.strip().split(':')[0], line.strip().split(':')[1]
        label_dict[folder] = label
        label_dict_res[label] = folder
print(label_dict)

N_CLASSES = 3
IMG_W = 224
IMG_H = IMG_W


def init_tf():
    global sess, pred, x
    #global count1, count2
    #count1 = 0
    #count2 = 0
    sess = tf.Session(config=config)
    #with gfile.FastGFile('./ncnn/save/model_1.3.pb', 'rb') as f:
    #with gfile.FastGFile('./ncnn/save/mobilenetv1_1.2.pb', 'rb') as f:
   # with gfile.FastGFile('./ncnn/save/mobilenetv1_quantize.pb', 'rb') as f:
    with gfile.FastGFile('./ncnn/save/model_quantize.pb', 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        sess.graph.as_default()
        tf.import_graph_def(graph_def, name='')
        print("graph:",tf.get_default_graph.__name__)

    # 获取输入tensor
    x = tf.get_default_graph().get_tensor_by_name("input_1:0")
    print("input:", x)
    # 获取预测tensor
    pred = tf.get_default_graph().get_tensor_by_name("sigmoid_out:0")  # mobilenet_v2
    print('load model done...')


def evaluate_image(img_dir,count1,count2):
    # read and process image
    image = cv2.imread(img_dir)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # pre-process the image for classification
    image = cv2.resize(image, (IMG_W, IMG_W))
    '''im_mean = np.mean(image)
    #stddev = np.mean(np.std(image, axis=(0, 1)))
    stddev = max(np.std(image), 1.0 / np.sqrt(IMG_W * IMG_W * 3))
    image1 = (image - im_mean) / stddev  # 代替tf.image.per_image_standardization'''
    image1 = np.array(image)#.astype(np.float32)
    image1 = np.expand_dims(image1, axis=0)
    #print(image1.dtype)
    prediction = sess.run(pred, feed_dict={x: image1})
    if prediction[0][0]>0.5:
        count1 += 1
    if prediction[0][1]>0.5:
        count2 += 1
    #print(prediction)
    max_index = np.argmax(prediction)
    pred_label = label_dict_res[str(max_index)]
    print(prediction[0],"%s, predict: %s(index:%d), prob: %f" % (img_dir, pred_label, max_index, prediction[0][max_index]))
    return count1,count2


if __name__ == '__main__':
    init_tf()
    #img_path = "D:/cat/Abyssinian_36.jpg"
    #evaluate_image(img_path)
    data_path = "D:/cat/cat1"
    #data_path = "D:/data/cat_dog"
    count=0
    count1=0
    count2=0
    for img in glob.glob(os.path.join(data_path, "*.jpg")):
        count += 1
        #print(img,count)
        count1, count2 = evaluate_image(img_dir=img, count1=count1, count2=count2)
    print((count1 / count)*100)
    print((count2 / count) * 100)


    '''label = os.listdir(data_path)
    for l in label:
        if os.path.isfile(os.path.join(data_path, l)):
            continue
        for img in glob.glob(os.path.join(data_path, l, "*.jpg")):
            evaluate_image(img_dir=img)'''
    sess.close()
