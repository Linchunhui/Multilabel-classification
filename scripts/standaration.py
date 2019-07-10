import tensorflow as tf
from keras.preprocessing import image
import numpy as np
import cv2
def init():
    global sess,x, mean, variance,sqrt,std,x1,x2,x3
    x = tf.placeholder(tf.float32, shape=[224, 224, 3])
    #x = tf.convert_to_tensor(x, name='image')
#    num_pixels = tf.reduce_prod(tf.shape(image))
    mean = tf.reduce_mean(x)
    variance = tf.reduce_mean(tf.square(x)-tf.square(mean))
    variance = tf.nn.relu(variance)
    sqrt=tf.sqrt(variance)
    std=tf.maximum(sqrt,(1.0/(224*224)))
    x1=tf.subtract(x,mean)
    x2=tf.div(x1,std)
    x3=tf.image.per_image_standardization(x)
    sess = tf.Session()

def gimage(path):
    img = image.load_img(path, target_size=(224, 224))
    image_array = image.img_to_array(img)
    mean1 = sess.run(mean, feed_dict={x: image_array})
    variance1 = sess.run(variance, feed_dict={x: image_array})
    sqrt1=sess.run(sqrt,feed_dict={x:image_array})
    std1=sess.run(std,feed_dict={x:image_array})
    x11 = sess.run(x1, feed_dict={x: image_array})
    x21 = sess.run(x2, feed_dict={x: image_array})
    x31 = sess.run(x3, feed_dict={x: image_array})
    return mean1,variance1,sqrt1,std1,x11,x21,x31

def std11(path):
    mean11=np.zeros([224,224,3])
    std=np.zeros([224,224,3])
    image=cv2.imread(path)
    image=cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
    mean11,std=cv2.meanStdDev(image,mean11,std)
    #mean11=(mean11[0][0]+mean11[1][0]+mean11[2][0])/3
    #std = (std[0][0] + std[1][0] + std[2][0]) / 3
    stda=max(std,(1.0/(224*224*3)))
    mean2=np.ones([224,224,3])
    #mean2=cv2.multiply()
    image1=image-mean11
    image1=image1/stda
    #image1=cv2.divide(image1,stda)
    return mean11,std,stda,image1,image
if __name__=="__main__":
    #init()
    path="D:/cat/cat1/ASDFruit_20181228_100.jpg"
    #m,v,s,s1,x11,x21,x31=gimage(path)
    #print(m)
    #print(s1)
    #print(x31)
   # print(x21==x31)
    mm,ss,ssa,i1,i=std11(path)
    print(mm)
    print(ssa)
    #print(img)
    print(i1)
    #print(x31==i1)
