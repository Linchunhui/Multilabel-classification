import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
import cv2
import numpy as np
import tensorflow as tf
import glob
import time
from tensorflow.python.platform import gfile

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # use gpu 0

label_dict, label_dict_res = {}, {}
with open("label.txt", 'r') as f:
    for line in f.readlines():
        folder, label = line.strip().split(':')[0], line.strip().split(':')[1]
        label_dict[folder] = label
        label_dict_res[label] = folder
print(label_dict)

N_CLASSES = len(label_dict)
IMG_W = 224
IMG_H = IMG_W


def init_tf():
    global sess, pred, x
    sess = tf.Session(config=config)
    #with gfile.FastGFile('./ncnn/save/model_1.3.pb', 'rb') as f:
    with gfile.FastGFile('./ncnn/save1/mobilenetv1.pb', 'rb') as f:
   # with gfile.FastGFile('./ncnn/save/mobilenetv1_quantize.pb', 'rb') as f:
    #with gfile.FastGFile('./ncnn/save/model_quantize.pb', 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        sess.graph.as_default()
        tf.import_graph_def(graph_def, name='')
    x = tf.get_default_graph().get_tensor_by_name("input_1:0")
    print("input:", x)
    pred = tf.get_default_graph().get_tensor_by_name("sigmoid_out:0")  # mobilenet_v2
    print('load model done...')

def evaluate_image(img_dir):
    # read and process image
    image = cv2.imread(img_dir)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # pre-process the image for classification
    image = cv2.resize(image, (IMG_W, IMG_W))
    im_mean = np.mean(image)
    stddev = max(np.std(image), 1.0 / np.sqrt(IMG_W * IMG_W * 3))
    image1 = (image - im_mean) / stddev  # 代替tf.image.per_image_standardization'''
    #image1 = np.array(image)
    image1 = np.expand_dims(image1, axis=0)
    #print(image1.dtype)
    prediction = sess.run(pred, feed_dict={x: image1})
    print(img_dir, prediction)
    #max_index = np.argmax(prediction)
    #pred_label = label_dict_res[str(max_index)]
    #print(prediction[0],"%s, predict: %s(index:%d), prob: %f" % (img_dir, pred_label, max_index, prediction[0][max_index]))


if __name__ == '__main__':
    init_tf()
    #img_path = "D:/cat/Abyssinian_36.jpg"
    #evaluate_image(img_path)
    data_path = "D:/cat/cat"
    #data_path = "D:/data/cat_dog"
    starttime = time.time()
    for img in glob.glob(os.path.join(data_path, "*.jpg")):
        evaluate_image(img_dir=img)
    endtime = time.time()
    print("time", endtime-starttime)
    print("fps", 1000/(endtime-starttime))
    sess.close()
