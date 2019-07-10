# coding:utf-8

import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.layers.python.layers import batch_norm

import tensorflow as tf
import tensorflow.contrib as tc

import numpy as np
import time


class MobileNetV1(object):
    def __init__(self, input,is_training=True, input_size=224):
        self.input=input
        self.input_size = input_size
        self.is_training = is_training
        self.normalizer = tc.layers.batch_norm
        self.bn_params = {'is_training': self.is_training}

        with tf.variable_scope('MobileNetV1'):
            self._build_model()

    def _build_model(self):
        i = 0
        with tf.variable_scope('init_conv'):
            self.conv1 = tc.layers.conv2d(self.input, num_outputs=32, kernel_size=3, stride=2,
                                          normalizer_fn=self.normalizer, normalizer_params=self.bn_params)
        # 1
        with tf.variable_scope('dconv_block{}'.format(i)):
            i += 1
            self.dconv1 = tc.layers.separable_conv2d(self.conv1, num_outputs=None, kernel_size=3, depth_multiplier=1,
                                                     normalizer_fn=self.normalizer, normalizer_params=self.bn_params)
            self.pconv1 = tc.layers.conv2d(self.dconv1, 64, 1, normalizer_fn=self.normalizer,
                                           normalizer_params=self.bn_params)
        # 2
        with tf.variable_scope('dconv_block{}'.format(i)):
            i += 1
            self.dconv2 = tc.layers.separable_conv2d(self.pconv1, None, 3, 1, 2,
                                                     normalizer_fn=self.normalizer, normalizer_params=self.bn_params)
            self.pconv2 = tc.layers.conv2d(self.dconv2, 128, 1, normalizer_fn=self.normalizer,
                                           normalizer_params=self.bn_params)
        # 3
        with tf.variable_scope('dconv_block{}'.format(i)):
            i += 1
            self.dconv3 = tc.layers.separable_conv2d(self.pconv2, None, 3, 1, 1,
                                                     normalizer_fn=self.normalizer, normalizer_params=self.bn_params)
            self.pconv3 = tc.layers.conv2d(self.dconv3, 128, 1, normalizer_fn=self.normalizer,
                                           normalizer_params=self.bn_params)
        # 4
        with tf.variable_scope('dconv_block{}'.format(i)):
            i += 1
            self.dconv4 = tc.layers.separable_conv2d(self.pconv3, None, 3, 1, 2,
                                                     normalizer_fn=self.normalizer, normalizer_params=self.bn_params)
            self.pconv4 = tc.layers.conv2d(self.dconv4, 256, 1, normalizer_fn=self.normalizer,
                                           normalizer_params=self.bn_params)
        # 5
        with tf.variable_scope('dconv_block{}'.format(i)):
            i += 1
            self.dconv5 = tc.layers.separable_conv2d(self.pconv4, None, 3, 1, 1,
                                                     normalizer_fn=self.normalizer, normalizer_params=self.bn_params)
            self.pconv5 = tc.layers.conv2d(self.dconv5, 256, 1, normalizer_fn=self.normalizer,
                                           normalizer_params=self.bn_params)
        # 6
        with tf.variable_scope('dconv_block{}'.format(i)):
            i += 1
            self.dconv6 = tc.layers.separable_conv2d(self.pconv5, None, 3, 1, 2,
                                                     normalizer_fn=self.normalizer, normalizer_params=self.bn_params)
            self.pconv6 = tc.layers.conv2d(self.dconv6, 512, 1, normalizer_fn=self.normalizer,
                                           normalizer_params=self.bn_params)
        # 7_1
        with tf.variable_scope('dconv_block{}'.format(i)):
            i += 1
            self.dconv71 = tc.layers.separable_conv2d(self.pconv6, None, 3, 1, 1,
                                                      normalizer_fn=self.normalizer, normalizer_params=self.bn_params)
            self.pconv71 = tc.layers.conv2d(self.dconv71, 512, 1, normalizer_fn=self.normalizer,
                                            normalizer_params=self.bn_params)
        # 7_2
        with tf.variable_scope('dconv_block{}'.format(i)):
            i += 1
            self.dconv72 = tc.layers.separable_conv2d(self.pconv71, None, 3, 1, 1,
                                                      normalizer_fn=self.normalizer, normalizer_params=self.bn_params)
            self.pconv72 = tc.layers.conv2d(self.dconv72, 512, 1, normalizer_fn=self.normalizer,
                                            normalizer_params=self.bn_params)
        # 7_3
        with tf.variable_scope('dconv_block{}'.format(i)):
            i += 1
            self.dconv73 = tc.layers.separable_conv2d(self.pconv72, None, 3, 1, 1,
                                                      normalizer_fn=self.normalizer, normalizer_params=self.bn_params)
            self.pconv73 = tc.layers.conv2d(self.dconv73, 512, 1, normalizer_fn=self.normalizer,
                                            normalizer_params=self.bn_params)
        # 7_4
        with tf.variable_scope('dconv_block{}'.format(i)):
            i += 1
            self.dconv74 = tc.layers.separable_conv2d(self.pconv73, None, 3, 1, 1,
                                                      normalizer_fn=self.normalizer, normalizer_params=self.bn_params)
            self.pconv74 = tc.layers.conv2d(self.dconv74, 512, 1, normalizer_fn=self.normalizer,
                                            normalizer_params=self.bn_params)
        # 7_5
        with tf.variable_scope('dconv_block{}'.format(i)):
            i += 1
            self.dconv75 = tc.layers.separable_conv2d(self.pconv74, None, 3, 1, 1,
                                                      normalizer_fn=self.normalizer, normalizer_params=self.bn_params)
            self.pconv75 = tc.layers.conv2d(self.dconv75, 512, 1, normalizer_fn=self.normalizer,
                                            normalizer_params=self.bn_params)
        # 8
        with tf.variable_scope('dconv_block{}'.format(i)):
            i += 1
            self.dconv8 = tc.layers.separable_conv2d(self.pconv75, None, 3, 1, 2,
                                                     normalizer_fn=self.normalizer, normalizer_params=self.bn_params)
            self.pconv8 = tc.layers.conv2d(self.dconv8, 1024, 1, normalizer_fn=self.normalizer,
                                           normalizer_params=self.bn_params)
        # 9
        with tf.variable_scope('dconv_block{}'.format(i)):
            i += 1
            self.dconv9 = tc.layers.separable_conv2d(self.pconv8, None, 3, 1, 1,
                                                     normalizer_fn=self.normalizer, normalizer_params=self.bn_params)
            self.pconv9 = tc.layers.conv2d(self.dconv9, 1024, 1, normalizer_fn=self.normalizer,
                                           normalizer_params=self.bn_params)
        with tf.variable_scope('global_max_pooling'):
            self.pool = tc.layers.max_pool2d(self.pconv9, kernel_size=7, stride=1)
        with tf.variable_scope('prediction'):
            self.output = tc.layers.conv2d(self.pool, 1000, 1, activation_fn=None)


class MobileNetV2(object):
    def __init__(self, input, num_classes=1000, is_training=True):
        self.input = input
        self.num_classes = num_classes
        self.is_training = is_training
        self.normalizer = tc.layers.batch_norm
        self.bn_params = {'is_training': self.is_training}

        with tf.variable_scope('MobileNetV2'):
            self._build_model()

    def _build_model(self):
        self.i = 0
        with tf.variable_scope('init_conv'):
            output = tc.layers.conv2d(self.input, 32, 3, 2,
                                      normalizer_fn=self.normalizer, normalizer_params=self.bn_params)
            # print(output.get_shape())
        self.output = self._inverted_bottleneck(output, 1, 16, 0)
        self.output = self._inverted_bottleneck(self.output, 6, 24, 1)
        self.output = self._inverted_bottleneck(self.output, 6, 24, 0)
        self.output = self._inverted_bottleneck(self.output, 6, 32, 1)
        self.output = self._inverted_bottleneck(self.output, 6, 32, 0)
        self.output = self._inverted_bottleneck(self.output, 6, 32, 0)
        self.output = self._inverted_bottleneck(self.output, 6, 64, 1)
        self.output = self._inverted_bottleneck(self.output, 6, 64, 0)
        self.output = self._inverted_bottleneck(self.output, 6, 64, 0)
        self.output = self._inverted_bottleneck(self.output, 6, 64, 0)
        self.output = self._inverted_bottleneck(self.output, 6, 96, 0)
        self.output = self._inverted_bottleneck(self.output, 6, 96, 0)
        self.output = self._inverted_bottleneck(self.output, 6, 96, 0)
        self.output = self._inverted_bottleneck(self.output, 6, 160, 1)
        self.output = self._inverted_bottleneck(self.output, 6, 160, 0)
        self.output = self._inverted_bottleneck(self.output, 6, 160, 0)
        self.output = self._inverted_bottleneck(self.output, 6, 320, 0)
        self.output = tc.layers.conv2d(self.output, 1280, 1, normalizer_fn=self.normalizer,
                                       normalizer_params=self.bn_params)
        self.output = tc.layers.avg_pool2d(self.output, 7)
        self.output = tc.layers.conv2d(self.output, self.num_classes, 1, activation_fn=None)
        self.output = tf.reshape(self.output, shape=[-1, self.num_classes], name="logit")

    def _inverted_bottleneck(self, input, up_sample_rate, channels, subsample):
        with tf.variable_scope('inverted_bottleneck{}_{}_{}'.format(self.i, up_sample_rate, subsample)):
            self.i += 1
            stride = 2 if subsample else 1
            output = tc.layers.conv2d(input, up_sample_rate * input.get_shape().as_list()[-1], 1,
                                      activation_fn=tf.nn.relu6,
                                      normalizer_fn=self.normalizer, normalizer_params=self.bn_params)
            output = tc.layers.separable_conv2d(output, None, 3, 1, stride=stride,
                                                activation_fn=tf.nn.relu6,
                                                normalizer_fn=self.normalizer, normalizer_params=self.bn_params)
            output = tc.layers.conv2d(output, channels, 1, activation_fn=None,
                                      normalizer_fn=self.normalizer, normalizer_params=self.bn_params)
            if input.get_shape().as_list()[-1] == channels:
                output = tf.add(input, output)
            return output


# small inception
def model4(x, N_CLASSES, is_trian=False):
    x = tf.contrib.layers.conv2d(x, 64, [5, 5], 1, 'SAME', activation_fn=tf.nn.relu)
    x = batch_norm(x, decay=0.9, updates_collections=None,
                   is_training=is_trian)  # ÑµÁ·½×¶Îis_traingingÉèÖÃÎªtrue,ÑµÁ·Íê±ÏºóÊ¹ÓÃÄ£ÐÍÊ±ÉèÖÃÎªfalse
    x = tf.contrib.layers.max_pool2d(x, [2, 2], stride=2, padding='SAME')

    x1_1 = tf.contrib.layers.conv2d(x, 64, [1, 1], 1, 'SAME', activation_fn=tf.nn.relu)  # 1X1 ºË
    x1_1 = batch_norm(x1_1, decay=0.9, updates_collections=None, is_training=is_trian)
    x3_3 = tf.contrib.layers.conv2d(x, 64, [3, 3], 1, 'SAME', activation_fn=tf.nn.relu)  # 3x3 ºË
    x3_3 = batch_norm(x3_3, decay=0.9, updates_collections=None, is_training=is_trian)
    x5_5 = tf.contrib.layers.conv2d(x, 64, [5, 5], 1, 'SAME', activation_fn=tf.nn.relu)  # 5x5 ºË
    x5_5 = batch_norm(x5_5, decay=0.9, updates_collections=None, is_training=is_trian)
    x = tf.concat([x1_1, x3_3, x5_5], axis=-1)  # Á¬½ÓÔÚÒ»Æð£¬µÃµ½64*3=192¸öÍ¨µÀ
    x = tf.contrib.layers.max_pool2d(x, [2, 2], stride=2, padding='SAME')

    x1_1 = tf.contrib.layers.conv2d(x, 128, [1, 1], 1, 'SAME', activation_fn=tf.nn.relu)
    x1_1 = batch_norm(x1_1, decay=0.9, updates_collections=None, is_training=is_trian)
    x3_3 = tf.contrib.layers.conv2d(x, 128, [3, 3], 1, 'SAME', activation_fn=tf.nn.relu)
    x3_3 = batch_norm(x3_3, decay=0.9, updates_collections=None, is_training=is_trian)
    x5_5 = tf.contrib.layers.conv2d(x, 128, [5, 5], 1, 'SAME', activation_fn=tf.nn.relu)
    x5_5 = batch_norm(x5_5, decay=0.9, updates_collections=None, is_training=is_trian)
    x = tf.concat([x1_1, x3_3, x5_5], axis=-1)
    x = tf.contrib.layers.max_pool2d(x, [2, 2], stride=2, padding='SAME')

    shp = x.get_shape()
    x = tf.reshape(x, [-1, shp[1] * shp[2] * shp[3]])  # flatten
    logits = tf.contrib.layers.fully_connected(x, N_CLASSES, activation_fn=None)  # output logist without softmax
    return logits


# 2conv + 3fc
def model2(images, batch_size, n_classes):
    '''Build the model
    Args:
        images: image batch, 4D tensor, tf.float32, [batch_size, width, height, channels]
    Returns:
        output tensor with the computed logits, float, [batch_size, n_classes]
    '''
    # conv1, shape = [kernel size, kernel size, channels, kernel numbers]

    with tf.variable_scope('conv1') as scope:
        weights = tf.get_variable('weights',
                                  shape=[3, 3, 3, 16],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32))
        biases = tf.get_variable('biases',
                                 shape=[16],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        conv = tf.nn.conv2d(images, weights, strides=[1, 1, 1, 1], padding='SAME')
        pre_activation = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(pre_activation, name=scope.name)

    # pool1 and norm1
    with tf.variable_scope('pooling1_lrn') as scope:
        pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                               padding='SAME', name='pooling1')
        norm1 = tf.nn.lrn(pool1, depth_radius=4, bias=1.0, alpha=0.001 / 9.0,
                          beta=0.75, name='norm1')

    # conv2
    with tf.variable_scope('conv2') as scope:
        weights = tf.get_variable('weights',
                                  shape=[3, 3, 16, 16],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32))
        biases = tf.get_variable('biases',
                                 shape=[16],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        conv = tf.nn.conv2d(norm1, weights, strides=[1, 1, 1, 1], padding='SAME')
        pre_activation = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(pre_activation, name='conv2')

    # pool2 and norm2
    with tf.variable_scope('pooling2_lrn') as scope:
        norm2 = tf.nn.lrn(conv2, depth_radius=4, bias=1.0, alpha=0.001 / 9.0,
                          beta=0.75, name='norm2')
        pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1],
                               padding='SAME', name='pooling2')

    # local3
    with tf.variable_scope('local3') as scope:
        reshape = tf.reshape(pool2, shape=[batch_size, -1])
        dim = reshape.get_shape()[1].value
        weights = tf.get_variable('weights',
                                  shape=[dim, 128],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.005, dtype=tf.float32))
        biases = tf.get_variable('biases',
                                 shape=[128],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        local3 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)

        # local4
    with tf.variable_scope('local4') as scope:
        weights = tf.get_variable('weights',
                                  shape=[128, 128],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.005, dtype=tf.float32))
        biases = tf.get_variable('biases',
                                 shape=[128],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        local4 = tf.nn.relu(tf.matmul(local3, weights) + biases, name='local4')

    # full connect
    with tf.variable_scope('softmax_linear') as scope:
        weights = tf.get_variable('softmax_linear',
                                  shape=[128, n_classes],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.005, dtype=tf.float32))
        biases = tf.get_variable('biases',
                                 shape=[n_classes],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        logits = tf.add(tf.matmul(local4, weights), biases, name='softmax_linear')

    return logits


if __name__ == '__main__':
    x = tf.placeholder(tf.float32, shape=[None, 224, 224, 3])
    model = MobileNetV2(x, num_classes=5, is_training=True)
    print
    "output size:"
    print
    model.output.get_shape()
    board_writer = tf.summary.FileWriter(logdir='./', graph=tf.get_default_graph())

    fake_data = np.ones(shape=(1, 224, 224, 3))

    sess_config = tf.ConfigProto(device_count={'GPU': 0})
    with tf.Session(config=sess_config) as sess:
        sess.run(tf.global_variables_initializer())

        cnt = 0
        for i in range(101):
            t1 = time.time()
            output = sess.run(model.output, feed_dict={x: fake_data})
            if i != 0:
                cnt += time.time() - t1
        print(cnt / 100)
