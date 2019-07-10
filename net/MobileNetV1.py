import tensorflow as tf
import tensorflow.contrib as tc

class MobileNetV1(object):
    def __init__(self, input,num_classes=1000,is_training=True, input_size=224):
        self.input=input
        self.input_size = input_size
        self.num_class=num_classes
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
            self.output = tc.layers.conv2d(self.pool, self.num_class, 1, activation_fn=None)
            self.output = tf.reshape(self.output, shape=[-1, self.num_class], name="logit")