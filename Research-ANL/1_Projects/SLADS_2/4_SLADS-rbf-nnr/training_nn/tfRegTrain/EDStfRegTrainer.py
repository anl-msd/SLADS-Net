import numpy as np
from matplotlib import pyplot as plt
import sys
import os
import tensorflow as tf



EDSvecB = np.load('EDSvecB.npy')
EDSvecM = np.load('EDSvecM.npy')


#plt.plot(EDSvecB[30,:])
#plt.show()
#
#plt.plot(EDSvecM[30,:])
#plt.show()


EDSlabelB = np.zeros(36)
EDSlabelM = np.ones(36)



d=12

EDStrain = np.vstack([EDSvecB[0:d,:], EDSvecM[0:d,:]])
#EDStrainlabel = np.hstack([EDSlabelB[0:d], EDSlabelM[0:d]])
EDStrainlabel = np.zeros((d*2, 2))
EDStrainlabel[0:d, 0] = 1
EDStrainlabel[d:d*2, 1] = 1

EDStest = np.vstack([EDSvecB[0+d:d+d,:], EDSvecM[0+d:d+d,:]])
#EDStestlabel = np.hstack([EDSlabelB[0+d:d+d], EDSlabelM[0+d:d+d]])
EDStestlabel = np.copy(EDStrainlabel)


EDStestErr = np.vstack([np.fliplr(EDSvecB[0+d:d+d,:]), np.fliplr(EDSvecM[0+d:d+d,:])])


noise1 = np.random.poisson(lam=2.0, size=(EDStrain.shape))
noise2 = np.random.poisson(lam=2.0, size=(EDStest.shape))
noise3 = np.random.poisson(lam=20.0, size=(EDStestErr.shape))


EDStrain = EDStrain + noise1
EDStest = EDStest + noise2
EDStestErr = EDStestErr + noise3



for i in range(0, 24):
    for j in range(0, 2042):
        if EDStestErr[i, j] > 20:
            EDStestErr[i, j] = 20
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


print(EDStrain.shape)
print(EDStrainlabel.shape)
print(EDStest.shape)
print(EDStestlabel.shape)


print(EDStestErr.shape)








x = tf.placeholder(tf.float32, shape=[None, 2042])   # x image data, has batch size not defined, dimension 28*28
y_ = tf.placeholder(tf.float32, shape=[None, 2])   # true distribution (one-hot vector)



def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev = 0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape = shape)
    return tf.Variable(initial)



#densely (fully) connected layer
W_ffc1 = weight_variable([2042, 100]) 
b_ffc1 = bias_variable([100])

W_ffc2 = weight_variable([100, 100]) 
b_ffc2 = bias_variable([100])

W_ffc3 = weight_variable([100, 100]) 
b_ffc3 = bias_variable([100])


h_ffc1 = (tf.matmul(x, W_ffc1) + b_ffc1) #

h_ffc2 = (tf.matmul(h_ffc1, W_ffc2) + b_ffc2) #

h_ffc3 = (tf.matmul(h_ffc2, W_ffc3) + b_ffc3) #

#W_ffco = weight_variable([8, 2])
#b_ffco = bias_variable([2])
#
#yy=(tf.matmul(h_ffc3, W_ffco) ) #+ b_ffco


yy = h_ffc3


y_tar = np.zeros(100)
for i in range(0, 100):
    y_tar[i] = i


# train and evaluation model
cross_entropy = tf.reduce_mean(tf.reduce_sum(tf.abs(yy - y_tar), reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
#correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
#accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
accuracy = tf.reduce_mean(tf.reduce_sum(tf.abs(yy - y_tar), reduction_indices=[1]))




init = tf.global_variables_initializer()

sess1 = tf.Session()

sess1.run(init)

saver = tf.train.Saver()





#sess1.run(tf.initialize_all_variables())

for i in range(1000):
    #batch = mnist.train.next_batch(4)
    #if i%10 == 0:
        
    train_accuracy = sess1.run(accuracy, feed_dict={x: EDStrain})
    print("step %d, training accuracy %g"%(i, train_accuracy))
    sess1.run(train_step, feed_dict={x: EDStrain})

    
x_test = EDStest
    
y_reg = sess1.run(yy, feed_dict={x: x_test})        
y_regnoise = sess1.run(yy, feed_dict={x: noise3})        




t=1
t1=t
t2=t+12
plt.figure()   
plt.plot(y_tar)
plt.plot(y_reg[t1,:])
#plt.figure()   
plt.plot(y_reg[t2,:])
#plt.figure()   
plt.plot(y_regnoise[t,:])




tf.add_to_collection('W_ffc1', W_ffc1)
tf.add_to_collection('b_ffc1', b_ffc1)
tf.add_to_collection('W_ffc2', W_ffc2)
tf.add_to_collection('b_ffc2', b_ffc2)
tf.add_to_collection('W_ffc3', W_ffc3)
tf.add_to_collection('b_ffc3', b_ffc3)

saver = tf.train.Saver()
saver.save(sess1, 'mymodel2')        
    























        