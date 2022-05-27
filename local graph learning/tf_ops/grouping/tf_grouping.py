import tensorflow as tf
from tensorflow.python.framework import ops
import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

def group_point(points, idx):
               
    B = points.shape[0] 
    B = int(B)
    B2 = idx.shape[1] 
    B2 = int(B2)
    new_points1 = []
    new_points3 = []
    for i in range(B):
        for j in range(B2):
            aa = idx[i,j,:]
            new_points1.append(tf.gather(points[i,:,:],axis=0,indices=aa))
        new_points2 = tf.stack(new_points1,axis=0) 
        new_points3.append(new_points2)
        new_points1 = []
    new_points4 = tf.stack(new_points3,axis=0) 
    return new_points4
def knn_point(k, xyz1, xyz2):
    b = xyz1.get_shape()[0].value
    n = xyz1.get_shape()[1].value
    c = xyz1.get_shape()[2].value
    m = xyz2.get_shape()[1].value
    
    xyz1 = tf.tile(tf.reshape(xyz1, (b,1,n,c)), [1,m,1,1])
    xyz2 = tf.tile(tf.reshape(xyz2, (b,m,1,c)), [1,1,n,1])
    dist = tf.reduce_sum((xyz1-xyz2)**2, -1)
    dist = tf.cast(dist,dtype=tf.int32) 
    outi, out = tf.nn.top_k(-dist,k)
    return out
