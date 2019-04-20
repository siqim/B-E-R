# -*- coding: utf-8 -*-
"""
Created on Sun Apr 14 17:11:55 2019

@author: msq96
"""

import pickle
import numpy as np
from tqdm import tqdm

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tensorboardX import SummaryWriter

from models import Sent2Class
from model_bak import VDCNN, SeNet2Class
from utils import save_model_optimizer, load_model_optimizer, try_mkdir, build_data


lr = 4e-4
wd = 4e-5
EPOCH = 10
batch_size = 128
test_freq_per_epoch = 5
patience = 1
freeze = False
dir = './data/datasets/yelp_review_full_csv/'

try_mkdir(dir+'models')
try_mkdir(dir+'models/logs')
try_mkdir(dir+'models/params')


#_, _, max_seq = build_data(dir=dir, filename='train.csv', max_seq=200, sep=' septor ', pad='zero_pad')
#build_data(dir=dir, filename='test.csv', max_seq=max_seq, sep=' septor ', pad='zero_pad')


print('Building data...')
X_train, y_train = pickle.load(open(dir+'train.pk', 'rb'))
X_test, y_test = pickle.load(open(dir+'test.pk', 'rb'))
y_train = y_train - 1
y_test = y_test - 1
num_classes = np.unique(y_train).shape[0]

print('Shuffing data...')
indexes = np.arange(X_train.shape[0])
np.random.shuffle(indexes)
X_train = X_train[indexes]
y_train = y_train[indexes]

indexes = np.arange(X_test.shape[0])
np.random.shuffle(indexes)
X_test = X_test[indexes]
y_test = y_test[indexes]


print('Spliting training set and testing set...')
num_batch_train = X_train.shape[0] // batch_size
num_batch_test = X_test.shape[0] // batch_size
num_batch_test = min(num_batch_test, 400)
test_per_n_batch = num_batch_train // test_freq_per_epoch
print('Test models every %d batches.'%test_per_n_batch)

print('Initializing models...')
model = SeNet2Class(num_classes=num_classes, embed_dim=300, path_embed_mat='./data/embeddings/embed_matrix.pk', freeze=False)
#model = TCNN(num_classes=num_classes, path_embed_mat='./data/embeddings/embed_matrix.pk', MAX_SENT_LEN=X_train.shape[1])
#model = VDCNN(n_classes=num_classes, path_embed_mat='./data/embeddings/embed_matrix.pk')
model.cuda()

print('Initializing optimizer...')
#optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

optimizer = torch.optim.Adam([
                                {'params': model.embedding.parameters(), 'lr': 4e-4, 'weight_decay':4e-5},
                                {'params': model.cnn.parameters(), 'lr': 1e-3, 'weight_decay': 4e-4}])

scheduler = ReduceLROnPlateau(optimizer, patience=patience, verbose=True)

print('Loading models and optimizer...')
model, optimizer, current_epoch, global_batch_counter_train, global_batch_counter_test = load_model_optimizer(model, optimizer, scheduler, dir)



print('Start training!')
writer = SummaryWriter(dir+'models/logs')

print('Changing learning rate and weight decay manually!')
for idx, param_group in enumerate(optimizer.param_groups, 1):

    if idx == 1:
        param_group['lr'] = 4e-6
        param_group['weight_decay'] = 0
    elif idx == 2:
        param_group['lr'] = 1e-5
        param_group['weight_decay'] = 0

    for idx, param_group in enumerate(optimizer.param_groups, 1):
        writer.add_scalar('epoch/lr_%d'%idx, param_group['lr'], global_batch_counter_train)

criterion = torch.nn.CrossEntropyLoss()
model.train()
for epoch in range(current_epoch, EPOCH):
    loss_train = 0

    for idx in tqdm(range(num_batch_train)):
        model.zero_grad()

        X_train_batch = X_train[idx*batch_size: (idx+1)*batch_size].cuda()
        y_train_batch = y_train[idx*batch_size: (idx+1)*batch_size].cuda()

        logit = model.forward(X_train_batch)
        loss = criterion(logit, y_train_batch)
        writer.add_scalar('loss_train_batch/loss_train_batch', loss, global_batch_counter_train)
        global_batch_counter_train += 1
        loss_train += loss.item()

        loss.backward()
        optimizer.step()

        if idx % test_per_n_batch == 0 and idx != 0:
            print('Start testing...')
            loss_test = 0
            total = 0
            correct = 0
            model.eval()
            with torch.no_grad():
                for test_idx in range(num_batch_test):

                    X_test_batch = X_test[test_idx*batch_size: (test_idx+1)*batch_size].cuda()
                    y_test_batch = y_test[test_idx*batch_size: (test_idx+1)*batch_size].cuda()

                    logit = model.forward(X_test_batch)
                    loss = criterion(logit, y_test_batch)
                    writer.add_scalar('loss_test_batch/loss_test_batch', loss, global_batch_counter_test)
                    global_batch_counter_test += 1
                    loss_test += loss.item()

                    _, predicted = torch.max(logit.data, 1)
                    total += y_test_batch.size(0)
                    correct += (predicted == y_test_batch).sum().item()


            indexes = np.arange(X_test.shape[0])
            np.random.shuffle(indexes)
            X_test = X_test[indexes]
            y_test = y_test[indexes]

            test_accuracy = correct/total
            writer.add_scalar('accuracy_test/accuracy_test', test_accuracy, global_batch_counter_test)
            print('[%d] epoch, [%.3f] training loss, [%.3f] testing loss, [%.3f] testing accuracy'
                  %(epoch, loss_train/test_per_n_batch, loss_test/test_idx, test_accuracy))

            print('Saving models and optimizer...')
            save_model_optimizer(model, optimizer, epoch, global_batch_counter_train, global_batch_counter_test, scheduler, dir)
            print('Saved!')

            loss_train = 0
            model.train()

    print('[%d] epoch finished!'%epoch)
    scheduler.step(loss_test/test_idx)

    for idx, param_group in enumerate(optimizer.param_groups, 1):
        writer.add_scalar('epoch/lr_%d'%idx, param_group['lr'], global_batch_counter_train)

    print('Saving models and optimizer...')
    save_model_optimizer(model, optimizer, epoch, global_batch_counter_train, global_batch_counter_test, scheduler, dir)
    print('Saved!')

    print('Shuffing training data for next epoch...')
    indexes = np.arange(X_train.shape[0])
    np.random.shuffle(indexes)
    X_train = X_train[indexes]
    y_train = y_train[indexes]


