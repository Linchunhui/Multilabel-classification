import tensorflow as tf
import tensorflow.contrib as tc

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
                                      activation_fn=tf.nn.relu,
                                      normalizer_fn=self.normalizer, normalizer_params=self.bn_params)
            output = tc.layers.separable_conv2d(output, None, 3, 1, stride=stride,
                                                activation_fn=tf.nn.relu,
                                                normalizer_fn=self.normalizer, normalizer_params=self.bn_params)
            output = tc.layers.conv2d(output, channels, 1, activation_fn=None,
                                      normalizer_fn=self.normalizer, normalizer_params=self.bn_params)
            if input.get_shape().as_list()[-1] == channels:
                output = tf.add(input, output)
            return output