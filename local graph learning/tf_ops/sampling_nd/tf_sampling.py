import tensorflow as tf
from tensorflow.python.framework import ops
import sys
import os
import numpy as np
    
def gather_point(points, idx):
               
    B = points.shape[0] 
    B = int(B)
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
        
    repeat_shape = list(idx.shape)
        
    repeat_shape[0] = 1
    batch_indices = tf.range(B,dtype=tf.int32)
        
    batch_indices = tf.reshape(batch_indices,(view_shape))
    batch_indices = tf.tile(batch_indices,repeat_shape)

    batch_indices=tf.cast(batch_indices,dtype=tf.int32)  
 
    new_points = [] 
    i=0
    for j in range(B):
        aa = idx[i,:]
        new_points.append(tf.gather(points[j,:,:],axis=0,indices=aa))
        i=i+1
        
    new_points = tf.stack(new_points,axis=0)  

    return new_points

def farthest_point_sample(npoint,xyz):

    batchsize = xyz.shape[0].value
    ndataset = xyz.shape[1].value
    dimension =xyz.shape[2].value  
    centroids = np.zeros([batchsize,npoint],dtype='int32')
      
    distance = np.ones([batchsize,ndataset])*1e10  
    farthest = np.random.randint(0,high=ndataset,size=(batchsize,))  
    batch_indices = np.arange(0,batchsize)
        
    for i in range(npoint):
            
        centroids0 = centroids[:,0:i]
        centroids1 = centroids[:,i+1:npoint]
        centroids2 = farthest 
        centroids2 = tf.expand_dims(centroids2,1)
        centroids = tf.concat([centroids0,centroids2,centroids1],1)   
        centroid = []
        j = 0 
        for i in range(batchsize):
            a = tf.gather(farthest,axis=0,indices=[i]) 
            centroid.append(tf.gather(xyz[j,:,:],axis=0,indices=a))
            j = j+1
                                
        centroid = tf.stack(centroid ,axis=0)
        centroid = tf.reshape(centroid,shape =[batchsize, 1, dimension])
        dist= tf.reduce_sum((xyz - centroid) ** 2,2) 
        dist = tf.cast(dist,dtype=tf.float64)
        distance = tf.cast(distance,dtype=tf.float64)
        distance = tf.where(dist < distance, dist, distance)
        farthest = tf.argmax(distance, 1)
        farthest = tf.cast(farthest,dtype=tf.int32)
    return centroids