import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class FlowNetS:
    def __init__(self, height, width):
        conv1 = layers.Conv2D(64, kernel_size=7, strides=2, padding='same', activation=tf.nn.relu,
                              input_shape=(height, width, 6), name='conv1')
        conv2 = layers.Conv2D(128, kernel_size=5, strides=2, padding='same', activation=tf.nn.relu, name='conv2')
        conv3 = layers.Conv2D(256, kernel_size=5, strides=2, padding='same', activation=tf.nn.relu, name='conv3')
        conv3_1 = layers.Conv2D(256, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu, name='conv3_1')
        conv4 = layers.Conv2D(512, kernel_size=3, strides=2, padding='same', activation=tf.nn.relu, name='conv4')
        conv4_1 = layers.Conv2D(512, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu, name='conv4_1')
        conv5 = layers.Conv2D(512, kernel_size=3, strides=2, padding='same', activation=tf.nn.relu, name='conv5')
        conv5_1 = layers.Conv2D(512, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu, name='conv5_1')
        conv6 = layers.Conv2D(1024, kernel_size=3, strides=2, padding='same', activation=tf.nn.relu, name='conv6')

        self.model = keras.Sequential([
            conv1,
            conv2,
            conv3,
            conv3_1,
            conv4,
            conv4_1,
            conv5,
            conv5_1,
            conv6,
        ])


path = 'D:\EduardoTayupanta\Documents\Librerias\FlowNet\checkpoints\FlowNetS'
file = 'flownet-S.ckpt-0'
trained_checkpoint_prefix = os.path.join(path, file)
net = FlowNetS(384, 1280)

names = [
    'conv1',
    'conv2',
    'conv3',
    'conv3_1',
    'conv4',
    'conv4_1',
    'conv5',
    'conv5_1',
    'conv6'
]
weights = {}
biases = {}
for name in names:
    weights['FlowNetS/' + name + '/weights:0'] = None
    biases['FlowNetS/' + name + '/biases:0'] = None

loaded_graph = tf.Graph()
with tf.compat.v1.Session(graph=loaded_graph) as sess:
    # Restore from checkpoint
    loader = tf.compat.v1.train.import_meta_graph(trained_checkpoint_prefix + '.meta')
    loader.restore(sess, trained_checkpoint_prefix)

    variables_names = [v.name for v in tf.compat.v1.trainable_variables()]
    values = sess.run(variables_names)
    for k, v in zip(variables_names, values):
        if k == 'FlowNetS/conv1/weights:0':
            weights[k] = v
        elif k == 'FlowNetS/conv2/weights:0':
            weights[k] = v
        elif k == 'FlowNetS/conv3/weights:0':
            weights[k] = v
        elif k == 'FlowNetS/conv3_1/weights:0':
            weights[k] = v
        elif k == 'FlowNetS/conv4/weights:0':
            weights[k] = v
        elif k == 'FlowNetS/conv4_1/weights:0':
            weights[k] = v
        elif k == 'FlowNetS/conv5/weights:0':
            weights[k] = v
        elif k == 'FlowNetS/conv5_1/weights:0':
            weights[k] = v
        elif k == 'FlowNetS/conv6/weights:0':
            weights[k] = v

        if k == 'FlowNetS/conv1/biases:0':
            biases[k] = v
        elif k == 'FlowNetS/conv2/biases:0':
            biases[k] = v
        elif k == 'FlowNetS/conv3/biases:0':
            biases[k] = v
        elif k == 'FlowNetS/conv3_1/biases:0':
            biases[k] = v
        elif k == 'FlowNetS/conv4/biases:0':
            biases[k] = v
        elif k == 'FlowNetS/conv4_1/biases:0':
            biases[k] = v
        elif k == 'FlowNetS/conv5/biases:0':
            biases[k] = v
        elif k == 'FlowNetS/conv5_1/biases:0':
            biases[k] = v
        elif k == 'FlowNetS/conv6/biases:0':
            biases[k] = v

    # all_vars = tf.compat.v1.get_collection(
    #     tf.compat.v1.GraphKeys.GLOBAL_VARIABLES
    # )
    #
    # print(all_vars)
    # print(all_vars[0], all_vars[1])

    # for v in all_vars:
    #     v_ = sess.run(v)
    #     print(v_.shape)

i = 0
for name_weights, name_biases in zip(weights, biases):
    l = []
    l.append(weights[name_weights])
    l.append(biases[name_biases])
    # net.model.layers[i].trainable = False
    net.model.layers[i].set_weights(l)
    i += 1

# for i in range(len(net.model.layers)):
#     print(net.model.layers[i].get_weights()[0].shape, net.model.layers[i].get_weights()[1].shape)
# with tf.device('/cpu:0'):

net.model.save('flownet_s.h5')
