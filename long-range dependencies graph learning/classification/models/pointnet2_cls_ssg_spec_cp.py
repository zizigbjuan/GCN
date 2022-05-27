import os
import sys
BASE_DIR = os.path.dirname(__file__)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../utils'))
import tensorflow as tf
import numpy as np
import tf_util
from pointnet_util import pointnet_sa_module 
from pointnet_util import pointnet_sa_module_spec 

from scipy.io import savemat

def placeholder_inputs(batch_size, num_point):
    pointclouds_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point, 3))
    labels_pl = tf.placeholder(tf.int32, shape=(batch_size))
    return pointclouds_pl, labels_pl

def get_model(point_cloud, is_training, bn_decay=None):
  
    batch_size = point_cloud.get_shape()[0].value
    num_point = point_cloud.get_shape()[1].value
    end_points = {}
    l0_xyz = tf.slice(point_cloud, [0,0,0], [-1,-1,3])
    l0_points = tf.slice(point_cloud, [0,0,0], [-1,-1,3])
    end_points['l0_xyz'] = l0_xyz

    l1_xyz, l1_points, l1_indices = pointnet_sa_module(l0_xyz, l0_points, npoint=512, radius=0.2, nsample=32, mlp=[64,64,128], mlp2=None, group_all=False, is_training=is_training, bn_decay=bn_decay, scope='layer1')
     
    #long-range dependencies graph learning    
    L2_xyz,L2_points, L2_indices = pointnet_sa_module_spec(l1_xyz, l1_points, npoint=128, radius=0.4, nsample=32, mlp=[128,256], mlp2=[256], group_all=True, is_training=is_training, bn_decay=bn_decay, scope='layer5' , knn=True, spec_conv_type = 'mlp', structure='spec' , useloc_covmat = True, pooling='max', csize = 2 )
    L3_xyz,L3_points, L3_indices = pointnet_sa_module_spec(L2_xyz, L2_points, npoint=32, radius=0.4, nsample=8, mlp=[256,512], mlp2=[512], group_all=True, is_training=is_training, bn_decay=bn_decay, scope='layer6' , knn=True, spec_conv_type = 'mlp', structure='spec' , useloc_covmat = True, pooling='max', csize = 2 )
    

    net = tf.reshape(L3_points, [batch_size, -1])
    net = tf_util.fully_connected(net, 256, bn=True, is_training=is_training, scope='fc1', bn_decay=bn_decay)
    net = tf_util.dropout(net, keep_prob=0.5, is_training=is_training, scope='dp1')
    net = tf_util.fully_connected(net, 128, bn=True, is_training=is_training, scope='fc2', bn_decay=bn_decay)
    net = tf_util.dropout(net, keep_prob=0.5, is_training=is_training, scope='dp2')
    net = tf_util.fully_connected(net, 10, activation_fn=None, scope='fc3') 
            
    return net,end_points
    

def get_loss(pred, label, end_points):

    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred, labels=label)
    classify_loss = tf.reduce_mean(loss)
    tf.summary.scalar('classify loss', classify_loss)
    return classify_loss


if __name__=='__main__':
    with tf.Graph().as_default():
        inputs = tf.zeros((32,1024,3))
        output, _ = get_model(inputs, tf.constant(True))
        print(output)
