import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
sys.path.append(os.path.join(ROOT_DIR, 'tf_ops/sampling_nd'))
sys.path.append(os.path.join(ROOT_DIR, 'tf_ops/grouping'))
sys.path.append(os.path.join(ROOT_DIR, 'tf_ops/3d_interpolation'))
from tf_sampling import farthest_point_sample, gather_point
from tf_grouping import group_point, knn_point

import tensorflow as tf
import numpy as np
import tf_util
from spec_graph_util import spec_conv2d
from spec_graph_util import spec_hier_cluster_pool


def sample_and_group(npoint, radius, nsample, xyz, points, tnet_spec=None, knn=False , use_xyz=True):

    new_xyz = gather_point(xyz, farthest_point_sample(npoint, xyz))

    idx = knn_point(nsample, xyz, new_xyz)

    grouped_xyz = group_point(xyz, idx) 
    grouped_xyz -= tf.tile(tf.expand_dims(new_xyz, 2), [1,1,nsample,1]) 
    
    if tnet_spec is not None:
        grouped_xyz = tnet(grouped_xyz, tnet_spec)

    if points is not None:
        grouped_points = group_point(points, idx) 
        if use_xyz:
            new_points = tf.concat([grouped_xyz, grouped_points], axis=-1) 
        else:
            new_points = grouped_points
    else:
        if use_xyz:
            new_points = grouped_xyz
        else:
            new_points = None

    return new_xyz, new_points, idx, grouped_xyz



def sample_and_group_all(xyz, points, use_xyz=True):

    batch_size = xyz.get_shape()[0].value
    nsample = xyz.get_shape()[1].value
    new_xyz = tf.constant(np.tile(np.array([0,0,0]).reshape((1,1,3)), (batch_size,1,1)),dtype=tf.float32) 
    idx = tf.constant(np.tile(np.array(range(nsample)).reshape((1,1,nsample)), (batch_size,1,1)))
    grouped_xyz = tf.reshape(xyz, (batch_size, 1, nsample, 3))
    if points is not None:
        if use_xyz:
            new_points = tf.concat([xyz, points], axis=2)
        else:
            new_points = points
        new_points = tf.expand_dims(new_points, 1)
    else:
        new_points = grouped_xyz
    return new_xyz, new_points, idx, grouped_xyz


def pointnet_sa_module(xyz, points, npoint, radius, nsample, mlp, mlp2, group_all, is_training, bn_decay, scope, bn=True, pooling='max', tnet_spec=None, knn=False, use_xyz=True ):
  
    with tf.variable_scope(scope) as sc:
       
        if group_all:
            nsample = xyz.get_shape()[1].value
            new_xyz, new_points, idx, grouped_xyz = sample_and_group_all(xyz, points, use_xyz)
        else:
            new_xyz, new_points, idx, grouped_xyz = sample_and_group(npoint, radius, nsample, xyz, points, tnet_spec, knn, use_xyz)

        
        for i, num_out_channel in enumerate(mlp):
            new_points = tf_util.conv2d(new_points, num_out_channel, [1,1],         
                                        padding='VALID', stride=[1,1],
                                        bn=bn, is_training=is_training,
                                        scope='conv_post_%d'%(i), bn_decay=bn_decay)
            
            
        if pooling == 'avg':
            new_points = tf_util.avg_pool2d(new_points, [1,nsample], stride=[1,1], padding='VALID', scope='avgpool1')
        elif pooling=='weighted_avg':
            with tf.variable_scope('weighted_avg1'):
                dists = tf.norm(grouped_xyz,axis=-1,ord=2,keep_dims=True)
                exp_dists = tf.exp(-dists * 5)
                weights = exp_dists/tf.reduce_sum(exp_dists,axis=2,keep_dims=True)
                new_points *= weights 
                new_points = tf.reduce_sum(new_points, axis=2, keep_dims=True)
        elif pooling=='max':
            new_points = tf.reduce_max(new_points, axis=[2], keep_dims=True)
        elif pooling=='min':
            new_points = tf_util.max_pool2d(-1*new_points, [1,nsample], stride=[1,1], padding='VALID', scope='minpool1')
        elif pooling=='max_and_avg':
            avg_points = tf_util.max_pool2d(new_points, [1,nsample], stride=[1,1], padding='VALID', scope='maxpool1')
            max_points = tf_util.avg_pool2d(new_points, [1,nsample], stride=[1,1], padding='VALID', scope='avgpool1')
            new_points = tf.concat([avg_points, max_points], axis=-1)
        elif pooling=='cluster_pool':
            new_points = spec_cluster_pool(new_points, pool_method = 'max')
        elif pooling=='hier_cluster_pool':
            new_points = spec_hier_cluster_pool(new_points, pool_method = 'max', csize = csize)
        elif pooling=='hier_cluster_pool_ablation':
            new_points = spec_hier_cluster_pool_ablation(new_points, pool_method = init_pooling, csize = csize, recurrence = r)
        
       
        if mlp2 is None: mlp2 = []
        for i, num_out_channel in enumerate(mlp2):
            new_points = tf_util.conv2d(new_points, num_out_channel, [1,1],
                                        padding='VALID', stride=[1,1],
                                        bn=bn, is_training=is_training,
                                        scope='conv_post_%d'%(i), bn_decay=bn_decay)
        new_points = tf.squeeze(new_points, [2]) 
        return new_xyz, new_points, idx

def pointnet_sa_module_spec(xyz, points, npoint, radius, nsample, mlp, mlp2, group_all, is_training, bn_decay, scope, bn=True, pooling='max', tnet_spec=None, knn=False, use_xyz=True , spec_conv_type = 'mlp', structure = 'spec', useloc_covmat = True, csize = None ):
 
     with tf.variable_scope(scope) as sc:       
        idx = farthest_point_sample(npoint, xyz)
        new_xyz = gather_point(xyz, idx)
        new_points = gather_point(points, idx)
               
        nsample = new_points.get_shape()[1].value
        batch_size = new_points.get_shape()[0].value
        feature = new_points.get_shape()[2].value
        
        local_cord = tf.reshape(new_points, (batch_size, 1, nsample, feature))
        new_points = tf.expand_dims(new_points, 1)  

        if useloc_covmat:
            local_cord = local_cord
        else:
            local_cord = None

        if structure == 'spec':
            mlp_spec = mlp
            new_points, UT = spec_conv2d_G(inputs = new_points, num_output_channels = mlp_spec, 
                                             local_cord = local_cord,
                                             bn=bn, is_training=is_training,
                                             scope='spec_conv%d'%(0), bn_decay=bn_decay)
       
        if pooling=='avg':
            new_points = tf_util.avg_pool2d(new_points, [1,nsample], stride=[1,1], padding='VALID', scope='avgpool1')
        elif pooling=='weighted_avg':
            with tf.variable_scope('weighted_avg1'):
                dists = tf.norm(grouped_xyz,axis=-1,ord=2,keep_dims=True)
                exp_dists = tf.exp(-dists * 5)
                weights = exp_dists/tf.reduce_sum(exp_dists,axis=2,keep_dims=True) 
                new_points *= weights 
                new_points = tf.reduce_sum(new_points, axis=2, keep_dims=True)
        elif pooling=='max':
            new_points = tf.reduce_max(new_points, axis=[2], keep_dims=True)
        elif pooling=='min':
            new_points = tf_util.max_pool2d(-1*new_points, [1,nsample], stride=[1,1], padding='VALID', scope='minpool1')
        elif pooling=='max_and_avg':
            avg_points = tf_util.max_pool2d(new_points, [1,nsample], stride=[1,1], padding='VALID', scope='maxpool1')
            max_points = tf_util.avg_pool2d(new_points, [1,nsample], stride=[1,1], padding='VALID', scope='avgpool1')
            new_points = tf.concat([avg_points, max_points], axis=-1)
        elif pooling=='hier_cluster_pool':
            new_points = spec_hier_cluster_pool(new_points, pool_method = 'max', csize = csize)
        elif pooling=='none':
            new_points = new_points

        
        if mlp2 is None: mlp2 = []
        for i, num_out_channel in enumerate(mlp2):
            new_points = tf_util.conv2d(new_points, num_out_channel, [1,1],
                                            padding='VALID', stride=[1,1],
                                            bn=bn, is_training=is_training,
                                            scope='conv_post_%d'%(i), bn_decay=bn_decay)                      
                    
        if pooling=='max':
            new_points = tf.squeeze(new_points, [2]) 
        elif pooling=='none':
            new_points = tf.squeeze(new_points, [1]) 
            
        return new_xyz, new_points, idx