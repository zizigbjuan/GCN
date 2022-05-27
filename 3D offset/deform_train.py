import os
import argparse
import numpy as np
from tqdm import tqdm
import pandas as pd
import joblib
from collections import OrderedDict
from datetime import datetime

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from utils import *
from scaled_mnist.dataset import ScaledMNIST
import scaled_mnist.archs as archs
import time

arch_names = archs.__dict__.keys()


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default=None,
                        help='model name: (default: arch+timestamp)')
    parser.add_argument('--arch', '-a', metavar='ARCH', default='ScaledMNISTNet',
                        choices=arch_names,
                        help='model architecture: ' +
                            ' | '.join(arch_names) +
                            ' (default: ScaledMNISTNet)')
    parser.add_argument('--deform', default=True, type=str2bool,
                        help='use deform conv')
    parser.add_argument('--modulation', default=True, type=str2bool,
                        help='use modulated deform conv')
    parser.add_argument('--min-deform-layer', default=3, type=int,
                        help='minimum number of layer using deform conv')
    parser.add_argument('--epochs', default=1, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--optimizer', default='SGD',
                        choices=['Adam', 'SGD'],
                        help='loss: ' +
                            ' | '.join(['Adam', 'SGD']) +
                            ' (default: Adam)')
    parser.add_argument('--lr', '--learning-rate', default=1e-2, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.5, type=float,
                        help='momentum')
    parser.add_argument('--weight-decay', default=1e-4, type=float,
                        help='weight decay')
    parser.add_argument('--nesterov', default=False, type=str2bool,
                        help='nesterov')

    args = parser.parse_args()

    return args
def get_batch(dataset,idxs, start_idx, end_idx):
    bsize = end_idx-start_idx
    cc = dataset.shape[1]
    batch_size = 10
    NUM_POINT = 512
    neighbor = 32

    batch_data = np.zeros((batch_size,1,32,NUM_POINT,3))
    for i in range(bsize):
        ps = dataset[idxs[i+start_idx]]        
        batch_data[i] = ps
    return batch_data

def get_batch0(label,idxs, start_idx, end_idx):
    bsize = end_idx-start_idx
    batch_size = 10
    batch_label = np.zeros((batch_size), dtype=np.int32)
    for i in range(bsize):
        cls = label[idxs[i+start_idx]]        
        batch_label[i] = cls
    return batch_label

def train(args, train_data,train_label, model, criterion, optimizer, epoch, scheduler=None):
    losses = AverageMeter()
    scores = AverageMeter()

    model.train()

    batch_size = 10
    num_batches = len(train_data)//batch_size    
    train_idxs = np.arange(0, len(train_data))
#     np.random.shuffle(train_idxs)
    train_data = torch.FloatTensor(train_data)
    a,b,c,d = train_data.shape
    train_data = train_data.reshape(a,1,c,b,d)
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = (batch_idx+1) * batch_size
        
        input = get_batch(train_data, train_idxs, start_idx, end_idx)
        target = get_batch0(train_label, train_idxs, start_idx, end_idx)
        
        input = torch.FloatTensor(input)
        target = torch.FloatTensor(target)
        target = target.long()        
        output,out = model(input)        
        
        loss = criterion(output, target) 
        
#         out0 = out
#         out0 = out0.permute(0,3,2,4,1)
#         out0 = out0.squeeze(4)
#         input0 = input   
#         input0 = input0.permute(0,3,2,4,1)
#         input0 = input0.squeeze(4)        
#         differences = out0-input0
#         d0 = torch.square(differences)            
#         d0 = torch.sum(d0, dim=3)            
#         d0 = (d0/0.2)**2
#         d0 = torch.min(d0,dim=1)                       
#         d0 = torch.mean(d0[0])
#         loss = loss+d0
        
        acc = accuracy(output, target)[0]

        losses.update(loss.item(), input.size(0))
        scores.update(acc.item(), input.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    log = OrderedDict([
        ('loss', losses.avg),
        ('acc', scores.avg),
    ])

    return log


def validate(args, val_data,val_label, model, criterion):
    losses = AverageMeter()
    scores = AverageMeter()

    model.eval()

    with torch.no_grad():
        batch_size = 10
        val_idxs = np.arange(0, len(val_data))
#         np.random.shuffle(val_idxs)
        num_batches = (len(val_data)+batch_size-1)//batch_size
        
        a,b,c,d = val_data.shape
        val_data = val_data.reshape(a,1,c,b,d)
        for batch_idx in range(num_batches):       
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx+1) * batch_size, len(val_data))
            bsize = end_idx - start_idx  
#             input = get_batch(val_data, val_idxs, start_idx, end_idx)
#             target = get_batch0(val_label, val_idxs, start_idx, end_idx) 

            input = val_data
            batch_label = np.zeros((3), dtype=np.int32)
            for i in range (3):
                batch_label[i] = val_label[i]
            target = batch_label
            input = torch.FloatTensor(input)
            target = torch.FloatTensor(target)
            target = target.long()
            
            output,out = model(input)

            loss = criterion(output, target)
            
#             out0 = out
#             out0 = out0.permute(0,3,2,4,1)
#             out0 = out0.squeeze(4)
#             input0 = input    ##[3, 1, 32, 1024, 3]
#             input0 = input0.permute(0,3,2,4,1)
#             input0 = input0.squeeze(4)        
#             differences = out0-input0
#             d0 = torch.square(differences)            
#             d0 = torch.sum(d0, dim=3)            
#             d0 = (d0/0.2)**2
#             d0 = torch.min(d0,dim=1)                       
#             d0 = torch.mean(d0[0])
#             loss = loss+d0
            
            acc = accuracy(output, target)[0]

            losses.update(loss.item(), input.size(0))
            scores.update(acc.item(), input.size(0))

    log = OrderedDict([
        ('loss', losses.avg),
        ('acc', scores.avg),
    ])

    return log


def main():
    args = parse_args()

    if args.name is None:
        args.name = '%s' %args.arch
        if args.deform:
            args.name += '_wDCN'
            if args.modulation:
                args.name += 'v2'
            args.name += '_c%d-4' %args.min_deform_layer

    if not os.path.exists('models/%s' %args.name):
        os.makedirs('models/%s' %args.name)

    print('Config -----')
    for arg in vars(args):
        print('%s: %s' %(arg, getattr(args, arg)))
    print('------------')

    with open('models/%s/args.txt' %args.name, 'w') as f:
        for arg in vars(args):
            print('%s: %s' %(arg, getattr(args, arg)), file=f)

    joblib.dump(args, 'models/%s/args.pkl' %args.name)

    criterion = nn.CrossEntropyLoss()

    cudnn.benchmark = True


    train_set = np.load(file=r'C:\Users\Desktop\train_data.npy')
    train_label = np.load(file=r'C:\Users\Desktop\train_label.npy')    

    test_set = np.load(file=r'C:\Users\Desktop\test_data.npy')
    test_label = np.load(file=r'C:\Users\Desktop\test_label.npy')    
    num_classes = 10

    model = archs.__dict__[args.arch](args, num_classes)
   
    if args.optimizer == 'Adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    elif args.optimizer == 'SGD':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr,
            momentum=args.momentum, weight_decay=args.weight_decay, nesterov=args.nesterov)

    log = pd.DataFrame(index=[], columns=[
        'epoch', 'lr', 'loss', 'acc', 'val_loss', 'val_acc'
    ])
    

    best_acc = 0
    epoch = 0
#     for epoch in range(args.epochs):
        
    t_start = time.process_time()
    t_wall  = time.time()
    print('Epoch [%d/%d]' %(epoch, args.epochs))


    # train for one epoch
#         train_log = train(args, train_set,train_label, model, criterion, optimizer, epoch)
    # evaluate on validation set
    val_log = validate(args, test_set,test_label, model, criterion)
    print('val_loss %.4f - val_acc %.4f'
        %(val_log['loss'], val_log['acc']))

#         print('loss %.4f - acc %.4f - val_loss %.4f - val_acc %.4f'
#             %(train_log['loss'], train_log['acc'], val_log['loss'], val_log['acc']))

    tmp = pd.Series([
        epoch,
        1e-1,         
        val_log['loss'],
        val_log['acc'],
    ], index=['epoch', 'lr', 'val_loss', 'val_acc'])

    log = log.append(tmp, ignore_index=True)
    log.to_csv('models/%s/log.csv' %args.name, index=False)

    if val_log['acc'] > best_acc:
        torch.save(model.state_dict(), 'models/%s/model.pth' %args.name)
        best_acc = val_log['acc']
        print("=> saved best model")
#         tmp = pd.Series([
#             epoch,
#             1e-1,
#             train_log['loss'],
#             train_log['acc'],
#             val_log['loss'],
#             val_log['acc'],
#         ], index=['epoch', 'lr', 'loss', 'acc', 'val_loss', 'val_acc'])

#         log = log.append(tmp, ignore_index=True)
#         log.to_csv('models/%s/log.csv' %args.name, index=False)

#         if val_log['acc'] > best_acc:
#             torch.save(model.state_dict(), 'models/%s/model.pth' %args.name)
#             best_acc = val_log['acc']
#             print("=> saved best model")

    print("best val_acc: %f" %best_acc)
    
    print('Execution time1: {:.2f}s'.format(time.process_time() - t_start))
    print('Execution time2: {:.2f}s'.format(time.time() - t_wall ))
        
if __name__ == '__main__':
    main()
