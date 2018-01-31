# -*- coding: utf-8 -*-
"""
Created on Sun Jan 28 14:32:23 2018
@author: alex.hall
"""
#TODO:
#Use pipelining API to allow for larger train set https://developers.googleblog.com/2017/09/introducing-tensorflow-datasets.html

#68% is current accuracy on valid set (basically guessing)

#also need to investigate why so many train set images are lost at point of load in

import os
import numpy as np 
import pandas as pd 
from glob import glob
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow as tf
import scipy
from sklearn.preprocessing import normalize

### parameters
res=128 #size to downsize images to
instance_multiplier=3 # scaling factor for train dataset - we have this number X min instnces of the smallest class of eachclass (some classes will require resampling)

#import metadata
xray_data=pd.read_csv("D:\\datasets\\xray\\Data_Entry_2017.csv\\Data_Entry_2017.csv")
labels_list=xray_data['Finding Labels'].str.cat(sep='|').split('|')
unique_labels=list(set(labels_list))


#split the image metadata into appropriate sets
test_set_data = xray_data.iloc[0:4998,].sample(2000) #IDEALLY LOSE THESE SAMPLES (ADDED TO DECREASE MEMORY USAGE)
valid_set_data = xray_data.iloc[5000:14997,].sample(2000)
train_set_data_all = xray_data.iloc[15000:,]

#get balanced training set
class_length=dict()
for label in unique_labels:
    class_length[label]=len(train_set_data_all[train_set_data_all["Finding Labels"].str.contains(label)])

#we will take this many instances of each class to build up the training set
instances_per_label= min(class_length.values())

#build up the training dataset metadata
train_set_data=pd.DataFrame()
for i in list(class_length.keys()):
    data=(train_set_data_all[train_set_data_all["Finding Labels"].str.contains(i)].sample(instance_multiplier*instances_per_label,replace=True))
    train_set_data=train_set_data.append(data)


#k hot encode the labels

def one_hot_encode(y_true,u_labels):
    y_out=np.zeros(shape=(len(y_true),len(u_labels)))
    for i in range(0,len(y_true)-1):
        for j in range(0,len(u_labels)-1):
            if(u_labels[j] in y_true[i]):
                y_out[i,j]=1
    return(y_out)

#function to reverse one hot encoding (used later for evaluation)
def one_hot_decode(instance):
    for i in range(0,len(instance)-1):
        if(int(instance[i])==1):
           return(unique_labels[i]) 

#print a sample image
#%matplotlib inline
"""
test_img=Image.open('C:\\Users\\alex.hall\\Documents\\datasets\\xray\\images_001\\images\\' + xray_data['Image Index'][10])
test_img_array=np.array(test_img)
print(np.shape(test_img_array))
plt.imshow(test_img)
"""
#### build sets with actual images
number_train_images=len(train_set_data)
number_test_images=len(test_set_data)
number_valid_images=len(valid_set_data) 

#shuffle the data
train_set_data.index=np.random.permutation(range(0,len(train_set_data)))
test_set_data.index=np.random.permutation(range(0,len(test_set_data)))
valid_set_data.index=np.random.permutation(range(0,len(valid_set_data)))

y=list()
x=list()
for i in range(0,int(number_train_images)):
    try:
        if(np.shape(np.array(Image.open('D:\\datasets\\xray\\train_set\\images\\' + train_set_data['Image Index'][i])))==(1024,1024)):
            raw_image=np.array(Image.open('D:\\datasets\\xray\\train_set\\images\\' + train_set_data['Image Index'][i]))
            resized_image=scipy.misc.imresize(raw_image,(res,res))
            x.append(resized_image)
            y.append(train_set_data.loc[i,'Finding Labels'])
    except:
        pass
    
x_train=np.array(x)
y_train=one_hot_encode(y,unique_labels)
######

y=list()
x=list()
for i in range(0,int(number_test_images)):
    try:
        if(np.shape(np.array(Image.open('D:\\datasets\\xray\\test_set\\images\\' + test_set_data['Image Index'][i])))==(1024,1024)):
            raw_image=np.array(Image.open('D:\\datasets\\xray\\test_set\\images\\' + test_set_data['Image Index'][i]))
            resized_image=scipy.misc.imresize(raw_image,(res,res))
            x.append(resized_image)
            y.append(test_set_data.loc[i,'Finding Labels'])
    except:
        pass
    
x_test=np.array(x)
y_test=one_hot_encode(y,unique_labels)
######

y=list()
x=list()
for i in range(0,int(number_valid_images)):
    try:
        if(np.shape(np.array(Image.open('D:\\datasets\\xray\\valid_set\\images\\' + valid_set_data['Image Index'][i])))==(1024,1024)):
            raw_image=np.array(Image.open('D:\\datasets\\xray\\valid_set\\images\\' + valid_set_data['Image Index'][i]))
            resized_image=scipy.misc.imresize(raw_image,(res,res))
            x.append(resized_image)
            y.append(valid_set_data.loc[i,'Finding Labels'])
    except:
        pass
    
x_valid=np.array(x)
y_valid=one_hot_encode(y,unique_labels)


######



test_set_indices=list()
valid_set_indices=list()
#use only single label images in test and valid sets
for i in range(0,len(y_test)-1):
    if(int(np.sum(y_test[[i]]))==1):
        test_set_indices.append(i)
        
for i in range(0,len(y_valid)-1):
    if(int(np.sum(y_valid[[i]]))==1):
        valid_set_indices.append(i)

y_test=y_test[test_set_indices]
x_test=x_test[test_set_indices]
y_valid=y_valid[valid_set_indices]
x_valid=x_valid[valid_set_indices]
y_test=np.reshape(y_test,(np.shape(y_test)[0],15))
y_train=np.reshape(y_train,(np.shape(y_train)[0],15))
y_valid=np.reshape(y_valid,(np.shape(y_valid)[0],15))

#now need to resize the images and normalize

"""
res=128 # resolution to downsize to
def downsize(arr,size):
    arr_new=np.empty(shape=(len(arr),size,size))
    for i in range(0,len(arr)-1):
        arr_new[i]=scipy.misc.imresize(arr[i],(res,res))
    return arr_new


x_test=downsize(x_test,res)
x_valid=downsize(x_valid,res)
x_train=downsize(x_train,res)
"""

def normalise(arr,resolution):
    arr_new=np.empty(shape=(len(arr),resolution,resolution))
    for i in range(0,len(arr)-1):
        arr_new[i]=normalize(arr[i])
    return arr_new

x_test=normalise(x_test,res)
x_valid=normalise(x_valid,res)
x_train=normalise(x_train,res)

##########

x = tf.placeholder(tf.float32,shape=[None,res,res])
y_true = tf.placeholder(tf.float32,shape=[None,15])
dropprob = tf.placeholder(tf.float32) #used for dropout

def init_weights(shape):
    init_random_dist = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(init_random_dist)

def init_bias(shape):
    init_bias_vals = tf.constant(0.1, shape=shape)
    return tf.Variable(init_bias_vals)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2by2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')

def convolutional_layer(input_x, shape):
    W = init_weights(shape)
    b = init_bias([shape[3]])
    return tf.nn.relu(conv2d(input_x, W) + b)

def normal_full_layer(input_layer, size):
    input_size = int(input_layer.get_shape()[1])
    W = init_weights([input_size, size])
    b = init_bias([size])
    return tf.matmul(input_layer, W) + b

x_image = tf.reshape(x,[-1,res,res,1])

convo_1 = convolutional_layer(x_image,shape=[4,4,1,128])
convo_1_pooling = max_pool_2by2(convo_1)

convo_2 = convolutional_layer(convo_1_pooling,shape=[4,4,128,256])
convo_2_pooling = max_pool_2by2(convo_2)

#convo_3 = convolutional_layer(convo_2_pooling,shape=[4,4,256,512])
#convo_3_pooling = max_pool_2by2(convo_3)


# 3 pooling layers with k size =2 and 3 convolutinal layers with stride=1; so so each conv+pool layer combination reduces size by a factor of 8
#, so (128/2)/2 = 32 . Note, we would need far more conv layers in a real
#life application but this would require a lot of runtime
# 128 then just comes from the output of the previous Convolution
convo_2_flat = tf.reshape(convo_2_pooling,[-1,32*32*256])

#placeholders for dropout
hold_prob = tf.placeholder(tf.float32)
full_layer_one = tf.nn.sigmoid(normal_full_layer(convo_2_flat,1024))
full_one_dropout = tf.nn.dropout(full_layer_one,keep_prob=hold_prob)
#full_layer_two = tf.nn.tanh(normal_full_layer(full_one_dropout,512))
#full_two_dropout = tf.nn.dropout(full_layer_two,keep_prob=hold_prob)
#full_layer_three = tf.nn.relu(normal_full_layer(full_layer_two,128))
#full_three_dropout = tf.nn.dropout(full_layer_three,keep_prob=hold_prob)


#output layer
y_pred = normal_full_layer(full_one_dropout,15)

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true,logits=y_pred))
optimizer = tf.train.AdagradOptimizer(learning_rate=0.001)
train = optimizer.minimize(cross_entropy)

import random
init = tf.global_variables_initializer()

steps = 1000
            
with tf.Session() as sess:
    
    sess.run(init)
    
    for i in range(steps):
        batch_indices=random.sample(range(0,np.shape(x_train)[0]), 10)
        batch_x  = x_train[batch_indices]
        batch_y = y_train[batch_indices]
        
        sess.run(train,feed_dict={x:batch_x,y_true:batch_y,hold_prob:0.5})
        
        # PRINT OUT A MESSAGE EVERY 100 STEPS
        if i%1 == 0:
            print('Currently on step {}'.format(i))
            print('Accuracy on validation set is:')
            
            # Test the Train Model
            matches = tf.equal(tf.argmax(y_pred,1),tf.argmax(y_true,1))
            
            acc = tf.reduce_mean(tf.cast(matches,tf.float32))
            
            print(sess.run(acc,feed_dict={x:x_valid,y_true:y_valid,hold_prob:1.0}))
            
            predictions=tf.argmax(y_pred,1)
            actuals=tf.argmax(y_valid,1) 
            
            #print(sess.run(predictions,feed_dict={x:x_test,y_true:y_test,hold_prob:1.0}))
            #print(sess.run(actuals,feed_dict={x:x_test,y_true:y_test,hold_prob:1.0}))
            
            c = tf.confusion_matrix(actuals, predictions)
            print(sess.run(c,feed_dict={x:x_valid,y_true:y_valid,hold_prob:1.0}))
           
            print('\n')
            
            

