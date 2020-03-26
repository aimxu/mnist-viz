#coding=utf-8

import tensorflow as tf
import numpy as np
import  matplotlib.pyplot as plt
import  os
from tensorflow.examples.tutorials.mnist.input_data import read_data_sets

# prefer gpu or cpu
PROCESSOR = 'GPU'

config = tf.ConfigProto()
if PROCESSOR == 'GPU':
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    # Option for GPU, allocating 50% GPU
    gpu_options = tf.GPUOptions(allow_growth=True, per_process_gpu_memory_fraction=0.5)
    config = tf.ConfigProto(gpu_options=gpu_options)
elif PROCESSOR == 'GPU':
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'


np.set_printoptions(suppress=True)

# prepare data
mnist = read_data_sets('./data', one_hot=True)
trainimg = mnist.train.images
trainlabel = mnist.train.labels
testimg = mnist.test.images
testlabel = mnist.test.labels


# pick a image randomly
n_rand = int(100000*np.random.rand()%10000)
input_image = testimg[n_rand:n_rand+1]
labels = trainlabel[n_rand:n_rand]+1



with tf.Session(config=config) as sess:
    # load and restore network and weights store in path ./model
    graph_path=os.path.abspath('./model/my-mnist-v1.0.meta')
    model=os.path.abspath('./model/')

    server = tf.train.import_meta_graph(graph_path)
    server.restore(sess,tf.train.latest_checkpoint(model))

    graph = tf.get_default_graph()

    x = graph.get_tensor_by_name('input_images:0')
    y = graph.get_tensor_by_name('input_labels:0')
    feed_dict={x:input_image,y:labels}


    # first conv and pool layer
    relu_1 = graph.get_tensor_by_name('relu_1:0')
    max_pool_1 = graph.get_tensor_by_name('max_pool_1:0')

    # second conv and pool layer
    relu_2 = graph.get_tensor_by_name('relu_2:0')
    max_pool_2 = graph.get_tensor_by_name('max_pool_2:0')

    # 2 fully connection layer
    fc_1 = graph.get_tensor_by_name('fc_1:0')
    fc_2 = graph.get_tensor_by_name('fc_2:0')

    # the output softmax layer
    f_softmax = graph.get_tensor_by_name('f_softmax:0')


    #----------------------------visualize each layer-------------------------------
    fig = plt.figure()

    ax = fig.add_subplot(5,1,1)
    ax.imshow(np.reshape(input_image, (28, 28)))
    plt.title('image')
    plt.axis('off')

    # conv1
    c1_relu = sess.run(relu_1,feed_dict)
    c1_tranpose = sess.run(tf.transpose(c1_relu,[3,0,1,2]))
    for i in range(16):
        ax = fig.add_subplot(5, 16, i+1+16)
        ax.imshow(c1_tranpose[i][0])
        plt.axis('off')
        if 16 / (i+1) == 2:
            plt.title('conv_1')

    # conv2
    c2_relu = sess.run(relu_2,feed_dict)
    c2_tranpose = sess.run(tf.transpose(c2_relu,[3,0,1,2]))
    for i in range(32):
        ax = fig.add_subplot(5, 32, i + 65)
        ax.imshow(c2_tranpose[i][0])
        plt.axis('off')
        if 32 / (i+1) == 2:
            plt.title('conv_2')
    # fc2
    c3_fc = sess.run(fc_2, feed_dict)
    c3_fc = np.tile(c3_fc, (4, 4))
    for i in range(1):
        ax = fig.add_subplot(5, 1, i + 4)
        ax.imshow(c3_fc)
        plt.axis('off')
    plt.title('fc_1')
    # output
    c4_sm = sess.run(f_softmax, feed_dict)
    for i in range(1):
        ax = fig.add_subplot(5, 1, i + 5)
        ax.imshow(c4_sm)
        plt.axis('off')
    plt.title('output')
    # show the whole image
    plt.show()

# print(sess.run(f_softmax, feed_dict))
# print(np.shape(c1_relu))
# print(np.shape(c1_tranpose))
# print(np.shape(c2_relu))
# print(np.shape(c2_tranpose))