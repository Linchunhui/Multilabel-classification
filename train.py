# coding:utf-8
from read_batch import *
from net.MobileNetV2 import MobileNetV2
from net.MobileNetV1 import MobileNetV1
from net.net import ShuffleNetV2
import tensorflow.contrib.slim as slim
import pandas as pd
sys.path.append("project/ShuffleNet")

train_dir = "D:/train"
logs_train_dir = './model_save/mobilenetv1/train4'
restore_log_dir = './model_save/mobilenetv1/train2'
init_lr = 0.001
BATCH_SIZE = 32
#train, train_label = get_files(train_dir)
t_list = pd.read_csv("D:/project/ShuffleNet/csv/img1.2.csv")
t_list['label'] = t_list['label'].apply(lambda x: [int(x[1]), int(x[3]), int(x[5])])
t_list['label'] = t_list['label'].apply(lambda x: [float(x[0]), float(x[1]), float(x[2])])
train = t_list['name'].tolist()
train_label = t_list['label'].tolist()
#print(t_list)
#train, train_label = t_list['name'], t_list['label']
one_epoch_step = len(train) / BATCH_SIZE
decay_steps = 25 * one_epoch_step
MAX_STEP = 50 * one_epoch_step
N_CLASSES = 3
IMG_W = 256
IMG_H = 256
img_w = 224
img_h = 224
CAPACITY = 1000
label_smoothing = 0.05
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # gpu编号
config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # 设置最小gpu使用量
batch_norm_params = {
    # Decay for the moving averages.
    'decay': 0.997,
    # epsilon to prevent 0s in variance.
    'epsilon': 0.001,
    # force in-place updates of mean and variance estimates
    'updates_collections': None,
    # Moving averages ends up in the trainable variables collection
    'variables_collections': [tf.GraphKeys.TRAINABLE_VARIABLES],
}

def main():
    global_step = tf.Variable(0, name='global_step', trainable=False)
    # label without one-hot
    batch_train, batch_labels = get_batch1(train,
                                          train_label,
                                          IMG_W,
                                          IMG_H,
                                          img_w,
                                          img_h,
                                          BATCH_SIZE,
                                          CAPACITY)

    #model = ShuffleNetV2(batch_train, N_CLASSES, model_scale=2.0, is_training=True)
    #model = MobileNetV2(batch_train, num_classes=N_CLASSES, is_training=True)
    #input, num_classes = 1000, is_training = True, input_size = 224):
    model = MobileNetV1(batch_train, num_classes=N_CLASSES, is_training=True)
    logits = model.output
    #logits = tf.reshape(logit, shape=[-1, N_CLASSES], name="logit")
    #logits=alexnet(x=batch_train, keep_prob=0.5, num_classes=N_CLASSES)
    #logits=model4(x=batch_train,N_CLASSES=N_CLASSES,is_trian=True)
    print(logits.get_shape())
    # loss
    #one_hot_labels = slim.one_hot_encoding(batch_labels, N_CLASSES)
    batch_labels = (1.0-label_smoothing)*batch_labels+label_smoothing/N_CLASSES
    cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=batch_labels)
    loss = tf.reduce_mean(cross_entropy, name='loss')
    tf.summary.scalar('train_loss', loss)

    # optimizer
    lr = tf.train.exponential_decay(learning_rate=init_lr, global_step=global_step, decay_steps=decay_steps,
                                    decay_rate=0.1)
    tf.summary.scalar('learning_rate', lr)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss, global_step=global_step)
    sess=tf.Session()
    #accuracy
    pred = tf.nn.sigmoid(logits)
    pred = tf.where(pred < 0.5, x=tf.zeros_like(pred), y=tf.ones_like(pred))
    print(pred)
    #labels = tf.cast(batch_labels, tf.int32)
    print(batch_labels)
    result = tf.equal(pred, batch_labels)
    result2 = tf.reduce_mean(tf.cast(result, tf.float32), 1)
    correct = tf.equal(result2, tf.ones_like(result2))
    correct = tf.cast(correct, tf.float32)
    accuracy = tf.reduce_mean(correct)
    tf.summary.scalar('train_acc', accuracy)

    '''correct1 = tf.nn.in_top_k(logits, batch_labels, 1)
    correct1 = tf.cast(correct1, tf.float16)
    accuracy1 = tf.reduce_mean(correct1)
    tf.summary.scalar('train_acc', accuracy1)'''

    summary_op = tf.summary.merge_all()
    #sess = tf.Session(config=config)
    train_writer = tf.summary.FileWriter(logs_train_dir, sess.graph)

    # saver = tf.train.Saver()
    var_list = tf.trainable_variables()
    g_list = tf.global_variables()
    bn_moving_vars = [g for g in g_list if 'moving_mean' in g.name]
    bn_moving_vars += [g for g in g_list if 'moving_variance' in g.name]
    var_list += bn_moving_vars
    saver = tf.train.Saver(var_list=var_list, max_to_keep=100)

    #print("batch labels :", sess.run(batch_labels))
    #saver = tf.train.Saver(max_to_keep=100)
    sess.run(tf.global_variables_initializer())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    saver.restore(sess, restore_log_dir+'/model.ckpt-250000')
    try:
        for step in range(int(MAX_STEP)):
            if coord.should_stop():
                break
            #_, learning_rate, tra_loss, tra_acc, tra_acc1 = sess.run([optimizer, lr, loss, accuracy,accuracy1])
            _, learning_rate, tra_loss,  tra_acc1 = sess.run([optimizer, lr, loss, accuracy])
            if step % 10 == 0:
                #print('Epoch %3d/%d, Step %6d/%d, lr %f, train loss = %.2f, train accuracy = %.2f%%, train accuracy1 = %.2f%%' % (
                print('Epoch %3d/%d, Step %6d/%d, lr %f, train loss = %.2f,train accuracy1 = %.2f%%' % (
                step / one_epoch_step, MAX_STEP / one_epoch_step, step, MAX_STEP, learning_rate, tra_loss,
                #tra_acc * 100.0,tra_acc1*100.0))
                tra_acc1 * 100.0))

            if step % 100 ==0:
                summary_str = sess.run(summary_op)
                train_writer.add_summary(summary_str, step)
            if step % 20000 == 0 or (step + 1) == MAX_STEP:
                print("weight saved ...")
                checkpoint_path = os.path.join(logs_train_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)
    except tf.errors.OutOfRangeError:
        print('Done training -- epoch limit reached')
    finally:
        coord.request_stop()

    coord.join(threads)
    sess.close()


if __name__ == '__main__':
    main()
