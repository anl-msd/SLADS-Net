import numpy as np
from matplotlib import pyplot as plt
import sys
import os
import tensorflow as tf



#EDSvecB = np.load('EDSvecB.npy')
#EDSvecM = np.load('EDSvecM.npy')
#
#
##plt.plot(EDSvecB[30,:])
##plt.show()
##
##plt.plot(EDSvecM[30,:])
##plt.show()
#
#
#EDSlabelB = np.zeros(36)
#EDSlabelM = np.ones(36)
#
#
#
#d=12
#
#EDStrain = np.vstack([EDSvecB[0:d,:], EDSvecM[0:d,:]])
##EDStrainlabel = np.hstack([EDSlabelB[0:d], EDSlabelM[0:d]])
#EDStrainlabel = np.zeros((d*2, 2))
#EDStrainlabel[0:d, 0] = 1
#EDStrainlabel[d:d*2, 1] = 1
#
#EDStest = np.vstack([EDSvecB[0+d:d+d,:], EDSvecM[0+d:d+d,:]])
##EDStestlabel = np.hstack([EDSlabelB[0+d:d+d], EDSlabelM[0+d:d+d]])
#EDStestlabel = np.copy(EDStrainlabel)
#
#
#EDStestErr = np.vstack([np.fliplr(EDSvecB[0+d:d+d,:]), np.fliplr(EDSvecM[0+d:d+d,:])])
#
#
#noise1 = np.random.poisson(lam=2.0, size=(EDStrain.shape))
#noise2 = np.random.poisson(lam=2.0, size=(EDStest.shape))
#noise3 = np.random.poisson(lam=20.0, size=(EDStestErr.shape))
#
#
#EDStrain = EDStrain + noise1
#EDStest = EDStest + noise2
#EDStestErr = EDStestErr + noise3
#
#
#
#for i in range(0, 24):
#    for j in range(0, 2042):
#        if EDStestErr[i, j] > 20:
#            EDStestErr[i, j] = 20
#        if EDStestErr[i, j] < 10:
#            EDStestErr[i, j] = 10


#for i in range(0, 24):
#    for j in range(0, 2042):
#        if EDStrain[i, j] < 10:
#            EDStrain[i, j] = 10
#        if EDStest[i, j] < 10:
#            EDStest[i, j] = 10



#plt.plot(EDStestErr[20,:])
#plt.show()
#EDSvalidB = np.load('/home/yzhang/Research-ANL/1_Projects/SLADS-Python-v4-EDS_ClfReg_20170407/ResultsAndData/EDSSpectra/NdFeB/Phase_0/EDSValidationB.npy')
#EDSvalidM = np.load('/home/yzhang/Research-ANL/1_Projects/SLADS-Python-v4-EDS_ClfReg_20170407/ResultsAndData/EDSSpectra/NdFeB/Phase_1/EDSValidationM.npy')
#
#
#EDStrainB = np.load('/home/yzhang/Research-ANL/1_Projects/SLADS-Python-v4-EDS_ClfReg_20170407/training_nn/EDSTrainingB.npy')
#EDStrainM = np.load('/home/yzhang/Research-ANL/1_Projects/SLADS-Python-v4-EDS_ClfReg_20170407/training_nn/EDSTrainingM.npy')
#
#
#EDStestB = np.load('/home/yzhang/Research-ANL/1_Projects/SLADS-Python-v4-EDS_ClfReg_20170407/training_nn/EDSTestingB.npy')
#EDStestM = np.load('/home/yzhang/Research-ANL/1_Projects/SLADS-Python-v4-EDS_ClfReg_20170407/training_nn/EDSTestingM.npy')







EDStrain = np.load('/home/yzhang/Research-ANL/1_Projects/SLADS-Python-v4-EDS_ClfReg_20170407/training_nn/EDStrain.npy')
EDStest = np.load('/home/yzhang/Research-ANL/1_Projects/SLADS-Python-v4-EDS_ClfReg_20170407/training_nn/EDStest.npy')
EDSlabel = np.load('/home/yzhang/Research-ANL/1_Projects/SLADS-Python-v4-EDS_ClfReg_20170407/training_nn/EDSlabel.npy')

num = EDStrain.shape[0]
dim = EDStrain.shape[1]




EDSvalid = np.load('/home/yzhang/Research-ANL/1_Projects/SLADS-Python-v4-EDS_ClfReg_20170413_EDSpaper/training_nn/EDSvalid.npy')
EDSvalid1 = EDSvalid[0:num/2,:]
EDSvalid2 = EDSvalid[num/2:num,:]
np.save('EDSvalid1.npy', EDSvalid1)
np.save('EDSvalid2.npy', EDSvalid2)



import matplotlib
x = np.arange(1.,2041.)/100
font = {'family' : 'normal', 'weight' : 'bold','size'   : 22} #
matplotlib.rc('font', **font)



plt.plot(x, np.transpose(EDStrain[10,:]), label='phase 1')
#plt.figure()
plt.plot(x, np.transpose(EDStrain[num/2+1, :]), label='phase 2')
plt.legend(bbox_to_anchor=(0.75, 0.98), loc=2, borderaxespad=0.)  
plt.xlabel('kEV',fontsize=20)
plt.ylabel('Counts',fontsize=20)












#sess = tf.InteractiveSession()

x = tf.placeholder(tf.float32, shape=[None, dim])   # x image data, has batch size not defined, dimension 28*28
y_ = tf.placeholder(tf.float32, shape=[None, 2])   # true distribution (one-hot vector)

#W = tf.Variable(tf.zeros([2042, 2]))
#b = tf.Variable(tf.zeros([2]))


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev = 0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape = shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides = [1,1,2,1], padding = 'SAME')
#
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1,1,10,1], strides = [1,1,2,1], padding = 'SAME')



# reshape x: ?, width, height, color channel
x_image = tf.reshape(x, [-1, 1, dim, 1])


#W_conv1 = tf.Variable(tf.truncated_normal([1,10,1,8], stddev=0.1), name='W_conv1')
#b_conv1 = tf.Variable(tf.constant(0.1, shape = [8]), name='b_conv1')
#
#W_conv2 = tf.Variable(tf.truncated_normal([1, 10, 8, 16], stddev=0.1), name='W_conv2')
#b_conv2 = tf.Variable(tf.constant(0.1, shape = [16]), name='b_conv2')


# first conv layer
W_conv1 = weight_variable([1,10,1,8]) # The convolution will compute 32 features for each 5x5 patch
b_conv1 = bias_variable([8])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) ) #+ b_conv1
h_pool1 = max_pool_2x2(h_conv1)

# second conv layer
W_conv2 = weight_variable([1, 10, 8, 16])
b_conv2 = bias_variable([16])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) ) #+ b_conv2
h_pool2 = max_pool_2x2(h_conv2)


size_hp = (int)(np.str(tf.Tensor.get_shape(h_pool2)[2]))

h_flat = tf.reshape(h_pool2, [-1, size_hp*16])


#densely (fully) connected layer
W_fc1 = weight_variable([size_hp*16, 100]) 
b_fc1 = bias_variable([100])

W_fc2 = weight_variable([100, 32]) 
b_fc2 = bias_variable([32])

W_fc3 = weight_variable([32, 8]) 
b_fc3 = bias_variable([8])



#W_fc1 = tf.Variable(tf.truncated_normal([size_hp*16, 32], stddev=0.1), name='W_fc1')
#b_fc1 = tf.Variable(tf.constant(0.1, shape = [32]), name='b_fc1')
#
#W_fc2 = tf.Variable(tf.truncated_normal([32, 32], stddev=0.1), name='W_fc2')
#b_fc2 = tf.Variable(tf.constant(0.1, shape = [32]), name='b_fc2')
#
#W_fc3 = tf.Variable(tf.truncated_normal([32, 8], stddev=0.1), name='W_fc3')
#b_fc3 = tf.Variable(tf.constant(0.1, shape = [8]), name='b_fc3')




h_fc1 = (tf.matmul(h_flat, W_fc1) ) #+ b_fc1

h_fc2 = (tf.matmul(h_fc1, W_fc2) ) #+ b_fc2

h_fc3 = (tf.matmul(h_fc2, W_fc3) ) #+ b_fc3




# dropout layer
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc3, keep_prob)

# readout layer
W_fco = weight_variable([8, 2])
b_fco = bias_variable([2])

#W_fco = tf.Variable(tf.truncated_normal([8, 2], stddev=0.1), name='W_fco')
#b_fco = tf.Variable(tf.constant(0.1, shape = [2]), name='b_fco')

# softmax layer
y_conv=tf.nn.softmax(tf.matmul(h_fc3, W_fco) ) #+ b_fco



# train and evaluation model
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


#sess = tf.InteractiveSession()


init = tf.global_variables_initializer()

sess = tf.Session()

sess.run(init)

saver = tf.train.Saver()


for i in range(100):
    #batch = mnist.train.next_batch(4)
    #if i%10 == 0:
    a1 = (sess.run(accuracy, feed_dict={x: EDStrain, y_: EDSlabel, keep_prob: 0.9}))
    #print(a1)    
    #train_accuracy = accuracy.eval(feed_dict={x: EDStrain, y_: EDStrainlabel, keep_prob: 0.9})
    #print("step %d, training accuracy %g"%(i, train_accuracy))
    print("step %d, training accuracy %g"%(i, a1))
    a2= (sess.run(train_step, feed_dict={x: EDStrain, y_: EDSlabel, keep_prob: 0.9}))
    
    
    
tf.add_to_collection('W_conv1', W_conv1)
tf.add_to_collection('b_conv1', b_conv1)
tf.add_to_collection('W_conv2', W_conv2)
tf.add_to_collection('b_conv2', b_conv2)
tf.add_to_collection('W_fc1', W_fc1)
tf.add_to_collection('b_fc1', b_fc1)
tf.add_to_collection('W_fc2', W_fc2)
tf.add_to_collection('b_fc2', b_fc2)
tf.add_to_collection('W_fc3', W_fc3)
tf.add_to_collection('b_fc3', b_fc3)
tf.add_to_collection('W_fco', W_fco)
tf.add_to_collection('b_fco', b_fco)

saver = tf.train.Saver()
saver.save(sess, '/home/yzhang/Research-ANL/1_Projects/SLADS-Python-v4-EDS_ClfReg_20170407/training_nn/tfClfTrain/tfClfmodel')       


#x_test = EDStest
##x_test = noise3   
#
#
#acc = sess.run(accuracy, feed_dict={x: x_test, y_: EDStestlabel, keep_prob: 1.0})
#t_label = sess.run(y_conv, feed_dict={x: x_test, y_: EDStestlabel, keep_prob: 1.0})
#
#print(t_label)
#print("test accuracy is: " + np.str(np.round(acc*100)) + "%")    
    


 
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    