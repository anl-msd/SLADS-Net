#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 10:31:08 2017

@author: yzhang
"""
import matplotlib





h_conv1_out = sess.run(h_conv1, feed_dict={x: EDStrain, y_: EDSlabel, keep_prob: 0.9})
h_conv2_out = sess.run(h_conv2, feed_dict={x: EDStrain, y_: EDSlabel, keep_prob: 0.9})
h_pool1_out = sess.run(h_pool1, feed_dict={x: EDStrain, y_: EDSlabel, keep_prob: 0.9})
h_pool2_out = sess.run(h_pool2, feed_dict={x: EDStrain, y_: EDSlabel, keep_prob: 0.9})
h_flat_out = sess.run(h_flat, feed_dict={x: EDStrain, y_: EDSlabel, keep_prob: 0.9})
h_fc1_out = sess.run(h_fc1, feed_dict={x: EDStrain, y_: EDSlabel, keep_prob: 0.9})
h_fc2_out = sess.run(h_fc2, feed_dict={x: EDStrain, y_: EDSlabel, keep_prob: 0.9})
h_fc3_out = sess.run(h_fc3, feed_dict={x: EDStrain, y_: EDSlabel, keep_prob: 0.9})
y_conv_out = sess.run(y_conv, feed_dict={x: EDStrain, y_: EDSlabel, keep_prob: 0.9})







x = np.arange(1.,2041.)/100
font = {'family' : 'normal','weight' : 'bold','size'   : 22}
matplotlib.rc('font', **font)


x_conv1 = np.linspace(1,2041, np.round(2041/2))/100








plt.figure()
plt.plot(x_conv1, h_conv1_out[10,0,:,1], label='phase 1')
plt.plot(x_conv1, h_conv1_out[13,0,:,1], label='phase 2')
plt.legend(bbox_to_anchor=(0.75, 0.98), loc=2, borderaxespad=0.)  
plt.xlabel('kEV',fontsize=20)
plt.ylabel('Counts',fontsize=20)

plt.figure()
plt.plot(x, h_conv2_out[10,0,:,1], label='phase 1')
plt.plot(x, h_conv2_out[13,0,:,1], label='phase 2')
plt.legend(bbox_to_anchor=(0.75, 0.98), loc=2, borderaxespad=0.)  
plt.xlabel('kEV',fontsize=20)
plt.ylabel('Counts',fontsize=20)

plt.figure()
plt.plot(x, h_pool1_out[10,0,:,1], label='phase 1')
plt.plot(x, h_pool1_out[13,0,:,1], label='phase 2')
plt.legend(bbox_to_anchor=(0.75, 0.98), loc=2, borderaxespad=0.)  
plt.xlabel('kEV',fontsize=20)
plt.ylabel('Counts',fontsize=20)

plt.figure()
plt.plot(x, h_pool2_out[10,0,:,1], label='phase 1')
plt.plot(x, h_pool2_out[13,0,:,1], label='phase 2')
plt.legend(bbox_to_anchor=(0.75, 0.98), loc=2, borderaxespad=0.)  
plt.xlabel('kEV',fontsize=20)
plt.ylabel('Counts',fontsize=20)

plt.figure()
plt.plot(x, h_flat_out[10,:], label='phase 1')
plt.plot(x, h_flat_out[13,:], label='phase 2')
plt.legend(bbox_to_anchor=(0.75, 0.98), loc=2, borderaxespad=0.)  
plt.xlabel('kEV',fontsize=20)
plt.ylabel('Counts',fontsize=20)

plt.figure()
plt.plot(x, h_fc1_out[10,:], label='phase 1')
plt.plot(x, h_fc1_out[13,:], label='phase 2')
plt.legend(bbox_to_anchor=(0.75, 0.98), loc=2, borderaxespad=0.)  
plt.xlabel('kEV',fontsize=20)
plt.ylabel('Counts',fontsize=20)

plt.figure()
plt.plot(x, h_fc2_out[10,:], label='phase 1')
plt.plot(x, h_fc2_out[13,:], label='phase 2')
plt.legend(bbox_to_anchor=(0.75, 0.98), loc=2, borderaxespad=0.)  
plt.xlabel('kEV',fontsize=20)
plt.ylabel('Counts',fontsize=20)

plt.figure()
plt.plot(x, h_fc3_out[10,:], label='phase 1')
plt.plot(x, h_fc3_out[13,:], label='phase 2')
plt.legend(bbox_to_anchor=(0.75, 0.98), loc=2, borderaxespad=0.)  
plt.xlabel('kEV',fontsize=20)
plt.ylabel('Counts',fontsize=20)





































