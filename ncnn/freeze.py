# coding:utf-8

import tensorflow as tf
from tensorflow.python.framework import graph_util
#from net import ShuffleNetV2
#from net.net import MobileNetV2
from net.MobileNetV1 import MobileNetV1
#from AlexNet import alexnet
#from model import model4
#import model

label_dict, label_dict_res = {}, {}
# 手动指定一个从类别到label的映射关系
with open("label.txt", 'r') as f:
    for line in f.readlines():
        folder, label = line.strip().split(':')[0], line.strip().split(':')[1]
        label_dict[folder] = label
        label_dict_res[label] = folder
print(label_dict)

IMG_W = 224
N_CLASSES = 3


def print_pb(output_graph_path):
    tf.reset_default_graph()  # 重置计算图
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    output_graph_def = tf.GraphDef()
    # 获得默认的图
    graph = tf.get_default_graph()
    with open(output_graph_path, "rb") as f:
        output_graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(output_graph_def, name="")
        # 得到当前图有几个操作节点
        print("%d ops in the final graph." % len(output_graph_def.node))
        # tensor_name = [tensor.name for tensor in output_graph_def.node]
        # print(tensor_name)
        print('---------------------------')
        for op in graph.get_operations():
            # print出tensor的name和值
            print(op.name, op.values())
    sess.close()


def save_new_ckpt(logs_train_dir, newckpt):
    x = tf.placeholder(tf.float32, shape=[None, IMG_W, IMG_W, 3], name="input_1")
    # predict
    logits = MobileNetV1(x, num_classes=N_CLASSES, is_training=False).output
    #print(logit)
    logit = tf.identity(logits,name="logit1")
    #print(logit1)
    #logits = tf.reshape(logit, shape=[-1, N_CLASSES], name="logit_2")
    #print(logits)
    #logit=alexnet(x=x, keep_prob=1, num_classes=N_CLASSES)
    #logit = model2(images=x, batch_size=1, n_classes=N_CLASSES)
    #logit = model4(x, N_CLASSES, is_trian=False)
    # logit = model.model2(x_4d, batch_size=1, n_classes=N_CLASSES)

    pred = tf.nn.sigmoid(logits, name="sigmoid_out")

    saver = tf.train.Saver()
    sess = tf.Session()
    saver.restore(sess, logs_train_dir)
    print("load model done...")
    saver.save(sess, newckpt)
    sess.close()
    print('save new model done...')


def freeze_graph(input_checkpoint, output_graph):
    # 指定输出的节点名称,该节点名称必须是原模型中存在的节点
    output_node_names = ["input_1","sigmoid_out"]
    saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=True)

    with tf.Session() as sess:
        saver.restore(sess, input_checkpoint)  # 恢复图并得到数据
        output_graph_def = graph_util.convert_variables_to_constants(  # 模型持久化，将变量值固定
            sess=sess,
            input_graph_def=sess.graph_def,  # 等于:sess.graph_def
            output_node_names=output_node_names)

        with tf.gfile.GFile(output_graph, "wb") as f:  # 保存模型
            f.write(output_graph_def.SerializeToString())  # 序列化输出
        print("%d ops in the final graph." % len(output_graph_def.node))  # 得到当前图有几个操作节点


if __name__ == "__main__":
    model_path = './model_save/mobilenetv1/train4/model.ckpt-140000'
    new_model_path = './model_save/mobilenetv1/train4/modelnew1.ckpt'
    pb_model = "./ncnn/save1/mobilenetv1.pb"
    save_new_ckpt(model_path, new_model_path)
    freeze_graph(new_model_path, pb_model)

