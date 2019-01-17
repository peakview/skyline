# -*- coding: utf-8 -*-
"""
Created on Jan 16 23:32:59 2018
Set a fixed std for 
@author: nn
"""

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input,BatchNormalization,Dense,Reshape,Flatten
import sys

class MyNormLayer(tf.keras.layers.Layer):
  def __init__(self, num_outputs, trainable=True):
    super(MyNormLayer, self).__init__()
    self.num_outputs = num_outputs
    self.trainable = trainable
    
  def build(self, input_shape):
    return
    
  def call(self, input):
    (normalized_tensor, mean, variance)=tf.keras.backend.normalize_batch_in_training(input,[1],[0],reduction_axes=(2))
    return normalized_tensor

def getData(Nsample,Ntx):
    datas = np.random.randn(Nsample,Ntx)*alpha+3
    return datas

Ntx=8

X_in = Input(shape=(Ntx,),name='X_in')
#
X_in_shaped=Reshape((Ntx,1))(X_in)
#Tx_norm = MyNormLayer((Ntx,),trainable=True)(X_in_shaped)
#center: If True, add offset of beta to normalized tensor. If False, beta is ignored.
#scale: If True, multiply by gamma. If False, gamma is not used. When the next layer is linear (also e.g. nn.relu), this can be disabled since the scaling will be done by the next layer.
Tx_norm = BatchNormalization(center=False,scale=False,axis=2,trainable=True,renorm=False)(X_in_shaped)
Tx_norm2=Flatten()(Tx_norm)
Tx_out = Dense(1) (Tx_norm2)

modelnorm = Model(inputs=[X_in],outputs=[Tx_norm])
model = Model(inputs=[X_in],outputs=[Tx_out])
print(model.summary())
model.compile(optimizer=tf.train.AdamOptimizer(), 
              loss='MSE',
              metrics=['accuracy'])

print(model.layers[2].get_weights())
#sys.exit(0)

alpha=2
Niteration = 40
means=np.zeros((Niteration,2))
stds=np.zeros((Niteration,2))
for ii in range(Niteration):
        
    Nsample = 1000
    datas = getData(Nsample,Ntx)
    y_train = np.ones(Nsample)#np.zeros(Nsample)#
    model.fit([datas],[y_train],epochs=5)
    
    datas = getData(Nsample,Ntx)
    y=modelnorm.predict([datas])
    
    #print(np.abs(y-datas).sum(axis=1).sum(axis=0))
    means[ii,:] = [np.mean(datas),np.mean(y)]
    stds[ii,:] = [np.std(datas),np.std(y)]
plt.plot(means[:,0],label='input mean')
plt.plot(means[:,1],label='output mean',linestyle='--')
plt.plot(stds[:,0],label='input std')
plt.plot(stds[:,1],label='output std',linestyle='--')
plt.title('stds[-1,:]'+str(stds[-3:-1,1]))
plt.legend()