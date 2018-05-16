




def tfRegress(x_test1):
    
    import numpy as np
    from matplotlib import pyplot as plt
    import sys
    import os
    import tensorflow as tf


    
    x_test = x_test1.reshape(1, 2042)
    
    
    
    sess1 = tf.Session()
    new_saver = tf.train.import_meta_graph('/home/yzhang/Research-ANL/EBSdata/tfRegTrain/mymodel2.meta')
    new_saver.restore(sess1, tf.train.latest_checkpoint('/home/yzhang/Research-ANL/EBSdata/tfRegTrain/./'))
    
    W_ffc1 = tf.get_collection('W_ffc1')[0]
    b_ffc1 = tf.get_collection('b_ffc1')[0]
    W_ffc2 = tf.get_collection('W_ffc2')[0]
    b_ffc2 = tf.get_collection('b_ffc2')[0]
    W_ffc3 = tf.get_collection('W_ffc3')[0]
    b_ffc3 = tf.get_collection('b_ffc3')[0]
    
    
    
    
    
    x = tf.placeholder(tf.float32, shape=[None, 2042])   # x image data, has batch size not defined, dimension 28*28
    #y_ = tf.placeholder(tf.float32, shape=[None, 100])   # true distribution (one-hot vector)
    
    
    
    def weight_variable(shape):
        initial = tf.truncated_normal(shape, stddev = 0.1)
        return tf.Variable(initial)
    
    def bias_variable(shape):
        initial = tf.constant(0.1, shape = shape)
        return tf.Variable(initial)
        
        
        
        
    h_ffc1 = (tf.matmul(x, W_ffc1) + b_ffc1) #
    
    h_ffc2 = (tf.matmul(h_ffc1, W_ffc2) + b_ffc2) #
    
    h_ffc3 = (tf.matmul(h_ffc2, W_ffc3) + b_ffc3) #
        
        
    yy = h_ffc3
    
    
    y_tar = np.zeros(100)
    for i in range(0, 100):
        y_tar[i] = i
      
        
        
    accuracy = tf.reduce_mean(tf.reduce_sum(tf.abs(yy - y_tar), reduction_indices=[1]))
    
    
    
    
    #init = tf.global_variables_initializer()
    #
    #sess1.run(init)
        
        
        
        
        
#    
#    noise3 = np.random.poisson(lam=20.0, size=(24, 2042))
#    
#    x_test = EDStest
        
    y_reg = sess1.run(yy, feed_dict={x: x_test})        
#    y_regnoise = sess1.run(yy, feed_dict={x: noise3})      
    
    
    
#    t=1
#    t1=t
#    t2=t+12
#    plt.figure()   
#    plt.plot(y_tar, label='target')
#    plt.plot(y_reg[t1,:], label='EDS-class-1')
#    #plt.figure()   
#    plt.plot(y_reg[t2,:], label='EDS-class-2')
#    #plt.figure()   
#    plt.plot(y_regnoise[t,:], label='Error case')
#    plt.legend(bbox_to_anchor=(0.01, 0.98), loc=2, borderaxespad=0.)  
    
    
    
    
    
    
    sig_true = np.var(np.abs(y_reg-y_tar))
#    
#    sig_false = np.var(np.abs(y_regnoise[t1,:]-y_tar))
    
    
    
    
    
    print(sig_true)
#    print(sig_false)
    return y_reg


























