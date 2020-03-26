#coding=utf-8
  
import  tensorflow as tf
import  matplotlib.pyplot as plt
import  numpy as np
from sklearn.metrics import classification_report
from tensorflow.examples.tutorials.mnist.input_data import read_data_sets
import time
import os

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



# hyperparameter
train_epochs = 5
batch_size = 100
drop_prob = 0.6
learning_rate=0.001

n_input = 784
n_output = 10

# initialize parameter weight and bias for each layer
def weight_init(shape):
    weight = tf.random_normal(shape=shape, stddev=0.1, dtype=tf.float32)
    return tf.Variable(weight)

def bias_init(shape):
    bias = tf.random_normal(shape=shape,stddev=0.1, dtype=tf.float32)
    return tf.Variable(bias)

# difine placeholder for data and label
# the name parameter is important for visualize
images_input = tf.placeholder(tf.float32,[None, n_input],name='input_images')
labels_input = tf.placeholder(tf.float32,[None, n_output],name='input_labels')


def fch_init(layer1,layer2,const=1):
    min = -const * (6.0 / (layer1 + layer2));
    max = -min
    weight = tf.random_uniform([layer1, layer2], minval=min, maxval=max, dtype=tf.float32)
    return tf.Variable(weight)

def conv2d(images,weight):
    return tf.nn.conv2d(images,weight,strides=[1,1,1,1],padding='SAME')

def max_pool2x2(images,name):
    return tf.nn.max_pool(images,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME',name=name)


x_input = tf.reshape(images_input,[-1,28,28,1])


# build the network
# first conv layer, conv kernel 3*3*16 using relu and dropout
# maxpooling 2*2
w1 = weight_init([3,3,1,16])
b1 = bias_init([16])
conv_1 = conv2d(x_input,w1) + b1
relu_1 = tf.nn.relu(conv_1,name='relu_1')
drop_out_1 = tf.nn.dropout(relu_1, keep_prob=drop_prob)
max_pool_1 = max_pool2x2(drop_out_1,name='max_pool_1')

# second conv layer, conv kernel 3*3*32 using relu and dropout
# maxpooling 2*2
w2 = weight_init([3,3,16,32])
b2 = bias_init([32])
conv_2 = conv2d(max_pool_1,w2) + b2
relu_2 = tf.nn.relu(conv_2,name='relu_2')
drop_out_2 = tf.nn.dropout(relu_2, keep_prob=drop_prob)
max_pool_2 = max_pool2x2(drop_out_2, name='max_pool_2')

# flatten the output, 7*7*32 ==> 1568
f_input = tf.reshape(max_pool_2,[-1,7*7*32])

# first fully connection layer
# 1568==>1024
f_w1= fch_init(7*7*32,1024)
f_b1 = bias_init([1024])
f_r1 = tf.matmul(f_input,f_w1) + f_b1
f_relu_r1 = tf.nn.relu(f_r1, name='fc_1')
f_dropout_r1 = tf.nn.dropout(f_relu_r1, keep_prob=drop_prob)

# second fully connection layer
# 1024==>128
f_w2 = fch_init(1024,128)
f_b2 = bias_init([128])
f_r2 = tf.matmul(f_dropout_r1,f_w2) + f_b2
f_relu_r2 = tf.nn.relu(f_r2, name='fc_2')
f_dropout_r2 = tf.nn.dropout(f_relu_r2,keep_prob=drop_prob)


# third fully connection layer and softmax output
# 128==>10
f_w3 = fch_init(128,10)
f_b3 = bias_init([10])
f_r3 = tf.matmul(f_dropout_r2,f_w3) + f_b3

y_pred = tf.nn.softmax(f_r3,name='f_softmax')

# 输出形状
print('C1 shape:', np.shape(drop_out_1))
print('P1 shape:', np.shape(max_pool_1))
print('C2 shape:', np.shape(drop_out_2))
print('P2 shape:', np.shape(max_pool_2))
print('F1 shape:', np.shape(f_dropout_r1))
print('F2 shape:', np.shape(f_dropout_r2))
print('SM shape:', np.shape(y_pred))

# using cross entropy for loss function
# and Adam Optimizer to minize the loss
cross_entry = tf.reduce_mean(-tf.reduce_sum(labels_input*tf.log(y_pred+1e-10)))
optimizer  = tf.train.AdamOptimizer(learning_rate).minimize(cross_entry)

# calculate loss and accuracy
arg1 = tf.argmax(labels_input,1)
arg2 = tf.argmax(y_pred,1)
cos = tf.equal(arg1,arg2)
acc = tf.reduce_mean(tf.cast(cos,dtype=tf.float32))

# prepare data
# using the old style
mnist = read_data_sets('./data', one_hot=True)
trainimg = mnist.train.images
trainlabel = mnist.train.labels
testimg = mnist.test.images
testlabel = mnist.test.labels

# start training
with tf.Session(config=config) as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    Cost = []
    Accuracy = []
    for i in range(train_epochs):
        start_time = time.time()
        print('epoch:', i+1)
        for j in range(int(60000/batch_size)):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            result,acc1,cross_entry_r,cos1,f_softmax1,relu_1_r= sess.run(
                [optimizer,acc,cross_entry,cos,y_pred,relu_1],
                feed_dict={
                    images_input:batch_xs,
                    labels_input:batch_ys})
            if j%100 == 0:
                Cost.append(cross_entry_r)
                Accuracy.append(acc1)
                print('step:', j, 'loss is:', cross_entry_r, ', accracy is:', acc1)
 
        use_time = time.time() - start_time
        print('epoch:', i+1, 'finished, using', use_time, 'S')

    # testing
    for i in range(10):
        batch_xs, batch_ys = mnist.test.next_batch(1000)
        arg1_r = sess.run(arg1, feed_dict={images_input: batch_xs, labels_input: batch_ys})
        arg2_r = sess.run(arg2, feed_dict={images_input: batch_xs, labels_input: batch_ys})
    print(classification_report(arg1_r, arg2_r))

    # save model to ./model
    saver = tf.train.Saver()
    saver.save(sess, './model/my-mnist-v1.0')

# plot the loss curve
fig1,ax1 = plt.subplots(figsize=(10,7))
plt.plot(Cost)
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Cost')
plt.title('Cross Loss')
plt.grid()
plt.show()

# plot the accuracy curve
fig7,ax7 = plt.subplots(figsize=(10,7))
plt.plot(Accuracy)
ax7.set_xlabel('Epochs')
ax7.set_ylabel('Accuracy Rate')
plt.title('Train Accuracy Rate')
plt.grid()
plt.show()








