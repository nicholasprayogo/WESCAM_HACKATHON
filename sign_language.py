#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  5 21:21:47 2018

@author: nick
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
df=pd.read_csv('sign_mnist_train.csv')
df=pd.DataFrame(df)
print(df)
print(df.head())
print(df.shape)
print(len(df.columns))

n=1
for i in range(25):
    image=np.array(df.iloc[n,1:len(df.columns)])
    A = df.iloc[n,0]
    B = np.reshape(image, (28, 28))
    print(B)
    plt.subplot(5,5,n)
    plt.imshow(B, 'gray', vmin=0, vmax=255)
    plt.title(chr(97+A))
    plt.gca().axes.get_xaxis().set_visible(False)
    plt.gca().get_yaxis().set_visible(False)
    n+=1


plt.tight_layout()
plt.show()
plt.savefig('sign_data.png')

def split():
    x=pd.DataFrame(df.iloc[:,1:len(df.columns)])
    y=pd.DataFrame(df.iloc[:,0])
    x_train, x_test, y_train, y_test=train_test_split(x,y,random_state=0)
    return(x_train, x_test, y_train, y_test)

x_train, x_test, y_train, y_test=split()

#preprocessing
from tflearn.layers.core import input_data, fully_connected, dropout
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.models.dnn import DNN
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation
from tflearn.layers.estimator import regression
import tensorflow as tf 

tf.reset_default_graph()

# Make sure the data is normalized
img_prep = ImagePreprocessing()
img_prep.add_featurewise_stdnorm()

# Create extra synthetic training data by flipping, rotating and blurring the
# images on our data set.
img_aug = ImageAugmentation()
img_aug.add_random_flip_leftright()
img_aug.add_random_rotation(max_angle=25.)
img_aug.add_random_blur(sigma_max=3.)

network= input_data(shape=[None, 28,28, 1], data_preprocessing=img_prep, data_augmentation=img_aug)
network = conv_2d(network, 32, 3, activation='relu')
network = max_pool_2d(network, 2)
network = conv_2d(network, 64, 3, activation='relu')
network = max_pool_2d(network, 2)
network = fully_connected(network, 784, activation='relu')
network = dropout(network, 0.5)
network = fully_connected(network, 26, activation='softmax')

network = regression(network, optimizer='adam', loss='categorical_crossentropy',learning_rate=0.001)

model = DNN(network, tensorboard_verbose=0, checkpoint_path='sign-language-classifier.tfl.ckpt')

model.fit(x_train, y_train, n_epoch=1, shuffle=True, validation_set=(x_test, y_test),
          show_metric=True, batch_size=96,
          snapshot_epoch=True, run_id='sign-language-classifier')

model.save("sign-language-classifier.tfl")

print(DNN.evaluate(x_test, y_test, batch_size=96))