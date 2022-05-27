# conding=utf-8
# # -*- coding:utf-8 -*- 
import argparse
import math
from datetime import datetime
import h5py
import numpy as np
import tensorflow as tf
import socket
import importlib
import os
import sys
import glob
import scipy.io as io
from sklearn.model_selection import train_test_split
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
import provider
import tf_util
import modelnet_dataset
from tensorflow.python import debug as tf_debug


parser = argparse.ArgumentParser()
parser.add_argument('--model', default='pointnet2_cls_ssg_spec_cpL',help='Model name [default:pointnet2_cls_ssg_spec_cpL]')
parser.add_argument('--subdir', default='', help='A sub dir that contains categoried models')
parser.add_argument('--log_dir', default='log', help='Log dir [default: log]')
parser.add_argument('--num_point', type=int, default=1024, help='Point Number [default: 1024]')
parser.add_argument('--max_epoch', type=int, default=250, help='Epoch to run [default: 251]')
parser.add_argument('--batch_size', type=int, default=10, help='Batch Size during training [default: 10]')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate [default: 0.001]')
parser.add_argument('--momentum', type=float, default=0.9, help='Initial learning rate [default: 0.9]')
parser.add_argument('--optimizer', default='adam', help='adam or momentum [default: adam]')
parser.add_argument('--decay_step', type=int, default=200000, help='Decay step for lr decay [default: 200000]')
parser.add_argument('--decay_rate', type=float, default=0.8, help='Decay rate for lr decay [default: 0.7]')
parser.add_argument('--normal', action='store_true', help='Whether to use normal information')
parser.add_argument('--debug', action='store_true', help='Whether to use debugger')
parser.add_argument('--eval_rotation', action='store_true', help='Whether to rotate in eval')
FLAGS = parser.parse_args()

EPOCH_CNT = 0

BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point
MAX_EPOCH = FLAGS.max_epoch
BASE_LEARNING_RATE = FLAGS.learning_rate
MOMENTUM = FLAGS.momentum
OPTIMIZER = FLAGS.optimizer
DECAY_STEP = FLAGS.decay_step
DECAY_RATE = FLAGS.decay_rate
SUBDIR = FLAGS.subdir
DEBUG = FLAGS.debug


sys.path.append(os.path.join(BASE_DIR , 'models' , SUBDIR))

MODEL = importlib.import_module(FLAGS.model) 
MODEL_FILE = os.path.join(BASE_DIR, 'models', FLAGS.subdir, FLAGS.model+'.py')


LOG_DIR_prefix = 'log'
EXP_prefix = ''
LOG_DIR = os.path.join(BASE_DIR,LOG_DIR_prefix,EXP_prefix,FLAGS.log_dir)
if not os.path.exists(LOG_DIR): os.mkdir(LOG_DIR)



os.system('cp %s %s' % (MODEL_FILE, LOG_DIR)) 
os.system('cp train.py %s' % (LOG_DIR)) 
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w')
LOG_FOUT.write(str(FLAGS)+'\n')

LOG_FOUT_max_record = open(os.path.join(LOG_DIR, 'log_max_record.txt'), 'w')
LOG_FOUT.write(str(FLAGS)+'\n')


BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
BN_DECAY_DECAY_STEP = float(DECAY_STEP)
BN_DECAY_CLIP = 0.99

HOSTNAME = socket.gethostname()

NUM_CLASSES = 10 


DATA_DIR = ROOT_DIR
DATA_PATH = os.path.join(DATA_DIR, 'data/modelnet10_normal_resampled')
TRAIN_DATASET = modelnet_dataset.ModelNetDataset(root=DATA_PATH, npoints=NUM_POINT, split='train', normal_channel=FLAGS.normal)
TEST_DATASET = modelnet_dataset.ModelNetDataset(root=DATA_PATH, npoints=NUM_POINT, split='test', normal_channel=FLAGS.normal)


def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)

def log_string_record(out_str , file):
    file.write(out_str+'\n')
    file.flush()
    print(out_str)

def get_learning_rate(batch):
    learning_rate = tf.train.exponential_decay(       
                        BASE_LEARNING_RATE, 
                        batch * BATCH_SIZE, 
                        DECAY_STEP,        
                        DECAY_RATE,        
                        staircase=True)
    learning_rate = tf.maximum(learning_rate, 0.00001)
    return learning_rate

def get_bn_decay(batch):
    bn_momentum = tf.train.exponential_decay(
                      BN_INIT_DECAY,
                      batch*BATCH_SIZE,
                      BN_DECAY_DECAY_STEP,
                      BN_DECAY_DECAY_RATE,
                      staircase=True)
    bn_decay = tf.minimum(BN_DECAY_CLIP, 1 - bn_momentum)
    return bn_decay

def train():
    with tf.Graph().as_default():

        pointclouds_pl, labels_pl = MODEL.placeholder_inputs(BATCH_SIZE, NUM_POINT)
        
        is_training_pl = tf.placeholder(tf.bool, shape=())

        batch = tf.Variable(0)
        bn_decay = get_bn_decay(batch)
        tf.summary.scalar('bn_decay', bn_decay) 

        pred, end_points = MODEL.get_model(pointclouds_pl,is_training_pl,bn_decay=bn_decay)                                     
           
        loss = MODEL.get_loss(pred, labels_pl, end_points)
        tf.summary.scalar('loss', loss)


        correct = tf.equal(tf.argmax(pred, 1), tf.to_int64(labels_pl))
        accuracy = tf.reduce_sum(tf.cast(correct, tf.float32)) / float(BATCH_SIZE) 
        tf.summary.scalar('accuracy', accuracy)

        print( "--- Get training operator")
            # Get training operator
        learning_rate = get_learning_rate(batch)
        tf.summary.scalar('learning_rate', learning_rate)
        if OPTIMIZER == 'momentum': 
            optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=MOMENTUM)
        elif OPTIMIZER == 'adam':
            optimizer = tf.train.AdamOptimizer(learning_rate)
        train_op = optimizer.minimize(loss, global_step=batch)
           
        saver = tf.train.Saver()

        config = tf.ConfigProto()
        
        config.allow_soft_placement = True
        config.log_device_placement = False
       
        merged = tf.summary.merge_all()
        sess = tf.Session(config=config)

        
        train_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'train'), sess.graph)
        test_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'test'), sess.graph)

        init = tf.global_variables_initializer()

        if FLAGS.debug:
            sess = tf_debug.LocalCLIDebugWrapperSession(sess)
            sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)

        sess.run(init)

        ops = {'pointclouds_pl': pointclouds_pl,
               'labels_pl': labels_pl,
               'is_training_pl': is_training_pl,
               'pred': pred,
               'loss': loss,
               'train_op': train_op,
               'merged': merged,
               'step': batch,
               'end_points': end_points
              }

        eval_acc_max_so_far = -1
        for epoch in range(MAX_EPOCH):
            log_string('**** EPOCH %03d ****' % (epoch))
            sys.stdout.flush()

            train_one_epoch(sess, ops, train_writer)                                               
            eval_acc_epoch = eval_one_epoch(sess, ops, test_writer)

            if epoch % 20 == 0:
                save_path = saver.save(sess, os.path.join(LOG_DIR, "model.ckpt"))
                log_string("Model saved in file: %s" % save_path)

            if eval_acc_epoch > eval_acc_max_so_far:
                eval_acc_max_so_far = eval_acc_epoch
                log_string_record('**** EPOCH %03d ****' % (epoch) , LOG_FOUT_max_record)
                log_string_record('eval accuracy: %f'% eval_acc_epoch , LOG_FOUT_max_record)
                save_path = saver.save(sess, os.path.join(LOG_DIR, "model_max_record.ckpt"))
                log_string_record("Model saved in file: %s" % save_path , LOG_FOUT_max_record)         
        
                    
def train_one_epoch(sess, ops, train_writer):
    """ ops: dict mapping from string to tf ops """
    is_training = True

    train_idxs = np.arange(0, len(TRAIN_DATASET))
    np.random.shuffle(train_idxs)
    num_batches = len(TRAIN_DATASET)//BATCH_SIZE

    log_string(str(datetime.now()))

    total_correct = 0
    total_seen = 0
    loss_sum = 0

    for batch_idx in range(num_batches):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = (batch_idx+1) * BATCH_SIZE
        batch_data, batch_label = get_batch(TRAIN_DATASET, train_idxs, start_idx, end_idx)
    
        aug_data = augment_batch_data(batch_data)
        
        aug_data = provider.random_point_dropout(aug_data)
        
        feed_dict = {ops['pointclouds_pl']: aug_data,
                     ops['labels_pl']: batch_label,
                     ops['is_training_pl']: is_training, }

        summary, step, _, loss_val, pred_val = sess.run([ops['merged'], ops['step'], ops['train_op'], ops['loss'], ops['pred']],
                                                        feed_dict=feed_dict)
                
        train_writer.add_summary(summary, step)
        pred_val = np.argmax(pred_val, 1)
        correct = np.sum(pred_val == batch_label)
        total_correct += correct
        total_seen += BATCH_SIZE
        loss_sum += loss_val

#         if (batch_idx+1)%50== 0:#################################################
    log_string(' --train accuracy %03d / %03d --' % (batch_idx+1, num_batches))
    log_string('mean loss: %f' % (loss_sum / 20))
    log_string('accuracy: %f' % (total_correct / float(total_seen)))
    total_correct = 0
    total_seen = 0
    loss_sum = 0
    
 
def eval_one_epoch(sess, ops, test_writer):
    global EPOCH_CNT
    is_training = False
    test_idxs = np.arange(0, len(TEST_DATASET))
    num_batches = (len(TEST_DATASET)+BATCH_SIZE-1)//BATCH_SIZE

    total_correct = 0
    total_seen = 0
    loss_sum = 0
    shape_ious = []
    total_seen_class = [0 for _ in range(NUM_CLASSES)]
    total_correct_class = [0 for _ in range(NUM_CLASSES)]

    log_string(str(datetime.now()))
    log_string('---- EPOCH %03d EVALUATION ----'%(EPOCH_CNT))

    for batch_idx in range(num_batches):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = min((batch_idx+1) * BATCH_SIZE, len(TEST_DATASET))
        bsize = end_idx - start_idx
        batch_data, batch_label = get_batch(TEST_DATASET, test_idxs, start_idx, end_idx)
               
        if FLAGS.eval_rotation:
            aug_data = augment_batch_data(batch_data)
        else:
            aug_data = batch_data

        feed_dict = {ops['pointclouds_pl']: aug_data,
                     ops['labels_pl']: batch_label,
                     ops['is_training_pl']: is_training}     
        summary, step, loss_val, pred_val = sess.run([ops['merged'], ops['step'],ops['loss'], ops['pred']], feed_dict=feed_dict)

        pred_val = np.argmax(pred_val, 1)
        correct = np.sum(pred_val[0:bsize] == batch_label[0:bsize])
        total_correct += correct
        total_seen += bsize
        loss_sum += (loss_val*float(bsize/BATCH_SIZE))
        for i in range(start_idx, end_idx):
            l = batch_label[i-start_idx]
            total_seen_class[l] += 1
            total_correct_class[l] += (pred_val[i-start_idx] == l)


    eval_acc = (total_correct / float(total_seen))
    eval_acc_class_avg = (np.mean(np.array(total_correct_class)/np.array(total_seen_class,dtype=np.float)))

    summary_acc = tf.Summary(value=[
                                    tf.Summary.Value(
                                                     tag='eval_acc', simple_value=float(eval_acc)),
                                    tf.Summary.Value(
                                                     tag='evl_acc_classavg', simple_value=float(eval_acc_class_avg))
                                    ])
    test_writer.add_summary(summary_acc, EPOCH_CNT)


    log_string('eval mean loss: %f' % (loss_sum / float(num_batches)))
    log_string('eval accuracy: %f'% (total_correct / float(total_seen)))
    log_string('eval avg class acc: %f' % (np.mean(np.array(total_correct_class)/np.array(total_seen_class,dtype=np.float))))
    EPOCH_CNT += 1
    return total_correct / float(total_seen)

def get_batch(dataset, idxs, start_idx, end_idx):
    bsize = end_idx-start_idx
    batch_data = np.zeros((BATCH_SIZE, NUM_POINT, dataset.num_channel()))
    batch_label = np.zeros((BATCH_SIZE), dtype=np.int32)
    for i in range(bsize):
        ps,cls = dataset[idxs[i+start_idx]]
        batch_data[i] = ps
        batch_label[i] = cls
    return batch_data, batch_label

def augment_batch_data(batch_data):

    if FLAGS.normal:
        rotated_data = provider.rotate_point_cloud_with_normal(batch_data)
        rotated_data = provider.rotate_perturbation_point_cloud_with_normal(rotated_data)
    else:
        rotated_data = provider.rotate_point_cloud(batch_data)
        rotated_data = provider.rotate_perturbation_point_cloud(rotated_data)

    jittered_data = provider.random_scale_point_cloud(rotated_data[:,:,0:3])
    jittered_data = provider.shift_point_cloud(jittered_data)
    jittered_data = provider.jitter_point_cloud(jittered_data)
    rotated_data[:,:,0:3] = jittered_data
    return rotated_data



if __name__ == "__main__":
    log_string('pid: %s'%(str(os.getpid())))
    train()
    LOG_FOUT.close()
