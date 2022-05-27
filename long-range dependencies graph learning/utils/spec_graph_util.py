import numpy as np
import tensorflow as tf
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




from tf_util import conv2d, batch_norm_for_conv2d, _variable_with_weight_decay

def corv_mat_setdiag_zero(adj_mat):
    
    in_shape = adj_mat.get_shape().as_list()

    adj_mat_set_diag = tf.zeros([in_shape[0], in_shape[1] , in_shape[2] ])
    adj_mat = tf.matrix_set_diag(adj_mat, adj_mat_set_diag)

    return adj_mat


def corv_mat_laplacian(adj_mat , flag_normalized = False):
    if not flag_normalized:
        D = tf.reduce_sum(adj_mat , axis = 3)
        D = tf.matrix_diag( D )
       
        L = D - adj_mat
    else:
        D = tf.reduce_sum(adj_mat , axis = 3)
        D_sqrt = tf.divide(1.0 , tf.sqrt(D) + 1e-8)
        D = tf.matrix_diag( D )
        D_sqrt = tf.matrix_diag( D_sqrt )

        L = tf.matmul(D_sqrt,
                      tf.matmul( D - adj_mat,
                               D_sqrt
                            )
                    )

    return L


def corv_mat_laplacian0(adj_mat , flag_normalized = False):
    if not flag_normalized:
        D = tf.reduce_sum(adj_mat , axis = 3)
        D = tf.matrix_diag( D )
       
        L = D - adj_mat
    else:
        D = tf.reduce_sum(adj_mat , axis = 3)
        D_sqrt = tf.divide(1.0 , tf.sqrt(D))
        D_sqrt = tf.matrix_diag( D_sqrt )

        I = tf.ones_like(D , dtype = tf.float32)
        I = tf.matrix_diag( I )
        L = I - tf.matmul(D_sqrt , tf.matmul( adj_mat , D_sqrt))

    return L

def corv_mat_diffusion(adj_mat):

    D = tf.reduce_sum(adj_mat , axis = 3)
    D = 1/D
    D = tf.matrix_diag( D )

    L = tf.matmul(D , adj_mat)

    return L


def cov_mat_k_nn_graph(adj_mat, k = 3):
  
    adj_sorted , adj_sort_ind = tf.nn.top_k(input = adj_mat , k = k, sorted=True)
    adj_thresh = adj_sorted[:,:,:, k - 1] 

    k_nn_adj_mat = tf.where( tf.less(adj_mat , tf.expand_dims( adj_thresh , axis = -1 ) ),
                        x = tf.zeros_like(adj_mat),
                        y = adj_mat,
                        name = 'k_nn_adj_mat'
                        )

    return k_nn_adj_mat



def get_adj_mat_dist_euclidean(local_cord , flag_normalized = False):
   
    in_shape = local_cord.get_shape().as_list()
  

    loc_matmul = tf.matmul(local_cord , local_cord, transpose_b=True)
    loc_norm = local_cord * local_cord 

    r = tf.reduce_sum(loc_norm , -1, keep_dims = True) 
    r_t = tf.transpose(r, [0,1,3,2])
    D = r - 2*loc_matmul + r_t

    D = tf.identity(D, name='adj_D')

    if flag_normalized:
        D_max = tf.reduce_max( tf.reshape(D , [in_shape[0] , in_shape[1] , in_shape[2] * in_shape[2]]) , axis = -1 )
        D_max = tf.expand_dims(D_max , -1)
        D_max = tf.expand_dims(D_max , -1)
        D_max = tf.tile(D_max , [1,1,in_shape[2],in_shape[2]])
        D = tf.divide(D , D_max + 1e-8)

   
    adj_mat = tf.exp(-D, name = 'adj_mat')

    return adj_mat



def get_adj_mat_cos( local_cord , flag_normalized = False, order = 1):
   
    in_shape = local_cord.get_shape().as_list()
   

    loc_matmul = tf.matmul(local_cord , local_cord, transpose_b=True, name = 'loc_matmul') 

    loc_norm = tf.norm(local_cord , axis = -1, keep_dims=True)
    loc_norm_matmul = tf.matmul(loc_norm , loc_norm, transpose_b=True, name = 'loc_norm_matmul') 
    D = tf.divide(loc_matmul , loc_norm_matmul + 1e-8 , name = 'cos_D')

  
    D = tf.exp(D*order)

    if flag_normalized:
        D_max = tf.reduce_max( tf.reshape(D , [in_shape[0] , in_shape[1] , in_shape[2] * in_shape[2]]) , axis = -1 )
        D_max = tf.expand_dims(D_max , -1)
        D_max = tf.expand_dims(D_max , -1)
        D_max = tf.tile(D_max , [1,1,in_shape[2],in_shape[2]])
        D = tf.divide(D , D_max + 1e-8)


    adj_mat = D

    return adj_mat



def spec_hier_cluster_pool(inputs , pool_method = 'max' , csize = 4, use_dist_adj = False, fast_approx = False, include_eig = False ):
    in_shape = inputs.get_shape().as_list()
    inputs_ = inputs
    
    K = in_shape[2]
    eig_2nd_saved = list()

    while(K > csize and K % csize == 0):
        
        if (not fast_approx) or (K == in_shape[2]):
           
            adj_mat = get_adj_mat_cos(inputs)  

            L = corv_mat_laplacian(adj_mat , flag_normalized = True)
            egval , egvect = tf.self_adjoint_eig(L)        
           
            ind = tf.constant( np.array([1]) ,dtype=tf.int32)
            partition_vect = tf.squeeze( tf.gather(egvect, ind , axis=-1) ) 
            eig_2nd_saved.append(partition_vect)

           
            eig_2nd_sorted , sort_ind = tf.nn.top_k(input = partition_vect , k=K, sorted=True)


  
            sort_ind = tf.reshape(sort_ind , [in_shape[0] * in_shape[1] , K ])
            
            inputs = tf.reshape(inputs , [in_shape[0] * in_shape[1] , K , in_shape[3]])
           
            inputs = gather_point(inputs, sort_ind)  
        else:
            inputs = tf.reshape(inputs , [in_shape[0] * in_shape[1] , K , in_shape[3]])
       
        inputs = tf.transpose(inputs , perm = [0,2,1])
        inputs = tf.reshape(inputs , [in_shape[0] * in_shape[1] , in_shape[3] ,int(K/csize) , csize ])
       
        if pool_method == 'max':
            inputs = tf.reduce_max(inputs , axis = -1)
            pool_method = 'avg'
        elif pool_method == 'avg':
            inputs = tf.reduce_mean(inputs , axis = -1)
            pool_method = 'max'
        
        K = int(K/csize)
        inputs = tf.reshape(tf.transpose(inputs, perm = [0,2,1]) ,
                            [in_shape[0] , in_shape[1] , K , in_shape[3]] )


   
    if pool_method == 'max':
        inputs = tf.reduce_max(inputs , axis = -2, keep_dims = True)
        pool_method = 'avg'
    elif pool_method == 'avg':
        inputs = tf.reduce_mean(inputs , axis = -2, keep_dims = True)
        pool_method = 'max'

    outputs = inputs


    return outputs





def weight_variable(shape, name=None , mean = 0, var = 0.1):
    initial = tf.truncated_normal_initializer(mean, var)
    var = tf.get_variable(name, shape, tf.float32, initializer=initial)

  
    return var




def spec_conv2d(inputs,
                num_output_channels,
                scope,
                nn_k = None,
                local_cord = None,
                use_xavier=True,
                stddev=1e-3,
                weight_decay=0.0,
                activation_fn=None,
                bn=False,
                bn_decay=None,
                is_training=None):


    in_shape = inputs.get_shape().as_list()
    
  
    W = get_adj_mat_dist_euclidean(local_cord[:,:,:,0:3] , flag_normalized = True)   
    W = tf.identity(W, name='adjmat')
    if nn_k is not None:
        num_neigh = nn_k
        W_knn = cov_mat_k_nn_graph(W, k = num_neigh )
    else:
        W_knn = W

    
    W_knn = corv_mat_setdiag_zero(W_knn)
    W_knn = tf.identity(W_knn, name='adjmat_knn')

    L = corv_mat_laplacian0(W_knn , flag_normalized = True)
    L = tf.identity(L, name='laplacian')


   

    egval , egvect = tf.self_adjoint_eig(L)
    U = egvect    
    
    UT = tf.transpose(U , perm = [0,1,3,2])
    
    inputs_fourier = tf.matmul( UT , inputs)

    filtered = inputs_fourier

   
    for i, num_out_channel in enumerate(num_output_channels):
        filtered = conv2d(filtered, num_out_channel, [1,1],
                          padding='VALID', stride=[1,1],
                          bn=False, is_training=is_training,
                          scope= scope + 'conv2d_%d'%(i),
                          bn_decay=bn_decay,
                          activation_fn = None)


    outputs = tf.matmul( U , filtered )

   
    outputs = batch_norm_for_conv2d(outputs, is_training, bn_decay=bn_decay, scope='bn_post_spec')
    outputs = tf.nn.relu(outputs)
    
    return outputs





def spec_conv2d_modul(inputs,
                num_output_channels,
                scope,
                nn_k = None,
                local_cord = None,
                use_xavier=True,
                stddev=1e-3,
                weight_decay=0.0,
                activation_fn=None,
                bn=False,
                bn_decay=None,
                is_training=None):


    in_shape = inputs.get_shape().as_list()
    
    
    W = get_adj_mat_dist_euclidean(local_cord[:,:,:,0:3] , flag_normalized = True)
    
    W = tf.identity(W, name='adjmat')

   
    if nn_k is not None:
        num_neigh = nn_k
        W_knn = cov_mat_k_nn_graph(W, k = num_neigh )
    else:
        W_knn = W

    
    W_knn = corv_mat_setdiag_zero(W_knn)
    W_knn = tf.identity(W_knn, name='adjmat_knn')

    L = corv_mat_laplacian0(W_knn , flag_normalized = True)
    L = tf.identity(L, name='laplacian')

    
    flag_use_svd = False
    if flag_use_svd:
        s, u, v  = tf.svd(                 
                          L,
                          compute_uv=True
                          )

        U = u
        UT = tf.transpose(v , perm = [0,1,3,2])
        egval = s

    else:
        egval , egvect = tf.self_adjoint_eig(L)
        U = egvect
        UT = tf.transpose(U , perm = [0,1,3,2])



    inputs_fourier = tf.matmul( UT , inputs)

   
    W_modulation = weight_variable(shape = (1,1, in_shape[-2]), name='spec_modulation' + scope , mean = 1.0, var = stddev)
    W_modulation = tf.matrix_diag( W_modulation )
    W_modulation = tf.tile(W_modulation , [ in_shape[0] , in_shape[1] , 1 , 1 ])
    filtered = tf.matmul( W_modulation , inputs_fourier)

    
    for i, num_out_channel in enumerate(num_output_channels):
        filtered = conv2d(filtered, num_out_channel, [1,1],
                          padding='VALID', stride=[1,1],
                          bn=False, is_training=is_training,
                          scope= scope + 'conv2d_%d'%(i),
                          bn_decay=bn_decay,
                          activation_fn = None)


    outputs = tf.matmul( U , filtered )

   
    outputs = batch_norm_for_conv2d(outputs, is_training, bn_decay=bn_decay, scope='bn_post_spec')
    outputs = tf.nn.relu(outputs)
    
    return outputs, UT

def spec_conv2d_G(inputs,
                num_output_channels,
                scope,
                nn_k = None,
                local_cord = None,
                use_xavier=True,
                stddev=1e-3,
                weight_decay=0.0,
                activation_fn=None,
                bn=False,
                bn_decay=None,
                is_training=None):


    in_shape = inputs.get_shape().as_list()
        
    adj_mat = get_adj_mat_cos(inputs)
    
    if nn_k is not None:
        num_neigh = nn_k
        W_knn = cov_mat_k_nn_graph(adj_mat, k = num_neigh )
    else:
        W_knn = adj_mat

    
    W_knn = corv_mat_setdiag_zero(W_knn)
    
    
    L,D = corv_mat_laplacian(adj_mat , flag_normalized = False)

    egval , egvect = tf.self_adjoint_eig(L)
    U = egvect
    UT = tf.transpose(U , perm = [0,1,3,2])    
    inputs_fourier = tf.matmul( UT , inputs)
    filtered = inputs_fourier
   
    for i, num_out_channel in enumerate(num_output_channels):
        filtered = conv2d(filtered, num_out_channel, [1,1],
                          padding='VALID', stride=[1,1],
                          bn=False, is_training=is_training,
                          scope= scope + 'conv2d_%d'%(i),
                          bn_decay=bn_decay,
                          activation_fn = None)


    outputs = tf.matmul( U , filtered )

   
    outputs = batch_norm_for_conv2d(outputs, is_training, bn_decay=bn_decay, scope='bn_post_spec')
    outputs = tf.nn.relu(outputs)
    
    return outputs, UT

def chebyshev(x,local_cord,Fout, K, scope,stddev=1e-3,weight_decay=0.0):

    N, M, H, Fin = x.get_shape()
    N, M, H, Fin = int(N), int(M),int(H), int(Fin)
            
    adj_mat = get_adj_mat_cos(x)  
    L = corv_mat_laplacian(adj_mat , flag_normalized = True)
    L = tf.identity(L, name='laplacian2')

    x0 = x
    x = tf.expand_dims(x0, 0) 

    def concat(x, x_):
        x_ = tf.expand_dims(x_, 0)  
        return tf.concat([x, x_], axis=0) 

    if K > 1:

        x1 = tf.matmul(L, x0) 
        x = concat(x, x1)
    for k in range(2, K):
        x2 = 2 * tf.matmul(L, x1) - x0  
        x = concat(x, x2)
        x0, x1 = x1, x2
    x = tf.transpose(x, perm=[1,2,3,4,0])  
    x = tf.reshape(x, [N * M * H, Fin * K])  

    W = weight_variable(shape = (Fin * K, Fout), name='spec_feature2'+scope, mean = 1.0, var = stddev)
    x = tf.matmul(x, W)  
    x = tf.reshape(x, [N, M, H, Fout]) 
    return x

