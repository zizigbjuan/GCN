
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

    # Local graph learning
    l1_xyz, l1_points, l1_indices = pointnet_sa_module(l0_xyz, l0_points, npoint=512, radius=0.2, nsample=32, mlp=[64,64,128], mlp2=None, group_all=False, is_training=is_training, bn_decay=bn_decay, scope='layer1')
    l2_xyz, l2_points, l2_indices = pointnet_sa_module_spec(l1_xyz, l1_points, npoint=128, radius=0.4, nsample=32, mlp=[128,256], mlp2=[256], group_all=False, is_training=is_training, bn_decay=bn_decay, scope='layer2' , knn=True, spec_conv_type = 'mlp', structure='spec' , useloc_covmat = True, pooling='max', csize = 2 )
    l3_xyz, l3_points, l3_indices = pointnet_sa_module_spec(l2_xyz, l2_points, npoint=32, radius=0.4, nsample=8, mlp=[256,512], mlp2=[512], group_all=False, is_training=is_training, bn_decay=bn_decay, scope='layer3' , knn=True, spec_conv_type = 'mlp', structure='spec' , useloc_covmat = True, pooling='max', csize = 2 )
    l4_xyz, l4_points, l4_indices = pointnet_sa_module_spec(l3_xyz, l3_points, npoint=None, radius=None, nsample=None, mlp=[512,1024], mlp2=None, group_all=True, is_training=is_training, bn_decay=bn_decay, scope='layer4', knn=True , spec_conv_type = 'mlp', structure='spec' , useloc_covmat = True, pooling='max')
    
    l1 = tf.reshape(l1_points, [batch_size, -1])
    l2 = tf.reshape(l2_points, [batch_size, -1])
    l3 = tf.reshape(l3_points, [batch_size, -1])
    l4 = tf.reshape(l4_points, [batch_size, -1])
    
    l1 = tf_util.fully_connected(l1, 512, bn=True, is_training=is_training, scope='fcL0', bn_decay=bn_decay)
    l1 = tf_util.dropout(l1, keep_prob=0.5, is_training=is_training, scope='dpL0')
    
    l2 = tf_util.fully_connected(l2, 512, bn=True, is_training=is_training, scope='fcL1', bn_decay=bn_decay)
    l2 = tf_util.dropout(l2, keep_prob=0.5, is_training=is_training, scope='dpL1')
    
    l3 = tf_util.fully_connected(l3, 512, bn=True, is_training=is_training, scope='fcL2', bn_decay=bn_decay)
    l3 = tf_util.dropout(l3, keep_prob=0.5, is_training=is_training, scope='dpL2')
    
    l4 = tf_util.fully_connected(l4, 512, bn=True, is_training=is_training, scope='fcL3', bn_decay=bn_decay)
    l4 = tf_util.dropout(l4, keep_prob=0.5, is_training=is_training, scope='dpL3')    
        
    emb = tf.stack([l1,l2,l3,l4],axis=1) 
    emb2 = tf.layers.dense(emb,10)

    emb2 = tf.nn.tanh(emb2)
    w = tf.layers.dense(emb2,10)
   
    beta = tf.nn.softmax(w,axis=1)
    net= tf.reduce_sum(beta*emb,axis=1)
    net = tf_util.fully_connected(net,1024, activation_fn=None, scope='fc0')
    net = tf_util.fully_connected(net,10, activation_fn=None, scope='fc1')
            
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
