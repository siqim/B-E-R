# -*- coding: utf-8 -*-
"""
Created on Sun Apr 14 17:11:55 2019

@author: msq96
"""


import pickle
from tqdm import tqdm

import torch
from tensorboardX import SummaryWriter
from pytorch_pretrained_bert.modeling import BertForSequenceClassification
from pytorch_pretrained_bert.optimization import BertAdam
from sklearn.metrics import f1_score

from utils import train, test, save_model_optimizer, load_model_optimizer, try_mkdir


LOG_NAME = 'finetune-20-one_epoch'
DEBUG = False
EPOCH = 30
save_freq_per_epoch = 0
save_freq_n_epoch = 5
assert bool(save_freq_n_epoch) != bool(save_freq_per_epoch)

dir = './data/datasets/yelp_review_polarity_csv/'

try_mkdir(dir+'models')
try_mkdir(dir+'models/logs')
try_mkdir(dir+'models/params')


print('Building data...')
label_ids, train_loader = pickle.load(open(dir+'train_loader.pk', 'rb'))
_, test_loader = pickle.load(open(dir+'test_loader.pk', 'rb'))


num_labels = len(label_ids)
num_train_batch = len(train_loader)
test_per_n_batch_one_epoch = num_train_batch // save_freq_per_epoch if save_freq_per_epoch!=0 else 0
num_train_optimization_steps = num_train_batch * EPOCH
batch_size_train = train_loader.batch_size
batch_size_test = test_loader.batch_size


print('Initializing models...')
#model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=num_labels)
model = BertForSequenceClassification.from_pretrained(pretrained_model_name_or_path=dir+'finetuned_lm/', num_labels=num_labels)
model.cuda()

print('Initializing optimizer...')
param_optimizer = list(model.named_parameters())
no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
optimizer = BertAdam(optimizer_grouped_parameters,
                     lr=1e-5,
                     warmup=0.1,
                     t_total=num_train_optimization_steps)

print('Loading models and optimizer...')
model, optimizer, current_epoch, global_batch_counter_train, global_batch_counter_test = load_model_optimizer(model, optimizer, dir)


print('Start training!')
writer = SummaryWriter(dir+'models/logs/%s/'%LOG_NAME)

loss_fct = torch.nn.CrossEntropyLoss()
model.train()
for epoch in range(current_epoch, EPOCH):
    loss_train = 0

    for idx, batch in enumerate(tqdm(train_loader), 1):

        loss, correct = train(batch, model, loss_fct, optimizer)

        writer.add_scalar('loss_train_batch/loss_train_batch', loss, global_batch_counter_train)
        writer.add_scalar('accuracy_train_batch/accuracy_train_batch', correct/batch[0].shape[0], global_batch_counter_train)

        global_batch_counter_train += 1
        loss_train += loss


        if (test_per_n_batch_one_epoch and (idx % test_per_n_batch_one_epoch == 0)) or (DEBUG and idx == 4 and test_per_n_batch_one_epoch):
            print('Start testing...')
            loss_test = 0
            total = 0
            correct = 0
            y_pred = []
            y_true = []
            logits = []
            model.eval()
            with torch.no_grad():
                for test_idx, batch in enumerate(tqdm(test_loader), 1):

                    loss, correct, total, y_pred_batch, y_true_batch, logit = test(batch, model, loss_fct, correct, total)
                    y_pred.extend(y_pred_batch)
                    y_true.extend(y_true_batch)
                    logits.extend(logit)

                    writer.add_scalar('loss_test_batch/loss_test_batch', loss, global_batch_counter_test)
                    global_batch_counter_test += 1
                    loss_test += loss

                    if DEBUG and test_idx == 4:
                        break

            test_mi_f1 = f1_score(y_true, y_pred, average='micro')
            test_ma_f1 = f1_score(y_true, y_pred, average='macro')
            test_accuracy = correct/total
            writer.add_scalar('accuracy_test/accuracy_test', test_accuracy, global_batch_counter_test)
            writer.add_scalar('accuracy_test/micro_f1_test', test_mi_f1, global_batch_counter_test)
            writer.add_scalar('accuracy_test/macro_f1_test', test_ma_f1, global_batch_counter_test)

            print('[%d] epoch, [%.3f] training loss, [%.3f] testing loss, [%.3f] testing accuracy'
                  %(epoch, loss_train/test_per_n_batch_one_epoch, loss_test/test_idx, test_accuracy))

            print('Saving models and optimizer...')
            save_model_optimizer(model, optimizer, epoch, global_batch_counter_train, global_batch_counter_test, dir)
            print('Saved!')

            loss_train = 0
            model.train()

        if DEBUG and idx == 4:
            break

    print('[%d] epoch finished!'%epoch)
    for id, param_group in enumerate(optimizer.param_groups, 1):
        writer.add_scalar('epoch/lr_%d'%id, param_group['lr'], global_batch_counter_train)

    if (save_freq_n_epoch and ((epoch+1) % save_freq_n_epoch == 0)):

        print('Start testing...')
        loss_test = 0
        total = 0
        correct = 0
        y_pred = []
        y_true = []
        logits = []
        model.eval()
        with torch.no_grad():
            for test_idx, batch in enumerate(tqdm(test_loader), 1):

                loss, correct, total, y_pred_batch, y_true_batch, logit = test(batch, model, loss_fct, correct, total)
                y_pred.extend(y_pred_batch)
                y_true.extend(y_true_batch)
                logits.extend(logit)

                writer.add_scalar('loss_test_batch/loss_test_batch', loss, global_batch_counter_test)
                global_batch_counter_test += 1
                loss_test += loss

                if DEBUG and test_idx == 4:
                    break

        test_mi_f1 = f1_score(y_true, y_pred, average='micro')
        test_ma_f1 = f1_score(y_true, y_pred, average='macro')
        test_accuracy = correct/total
        writer.add_scalar('accuracy_test/accuracy_test', test_accuracy, global_batch_counter_test)
        writer.add_scalar('accuracy_test/micro_f1_test', test_mi_f1, global_batch_counter_test)
        writer.add_scalar('accuracy_test/macro_f1_test', test_ma_f1, global_batch_counter_test)
        print('[%d] epoch, [%.3f] training loss, [%.3f] testing loss, [%.3f] testing accuracy'
              %(epoch, loss_train/len(train_loader), loss_test/test_idx, test_accuracy))

        print('Saving models and optimizer...')
        save_model_optimizer(model, optimizer, epoch, global_batch_counter_train, global_batch_counter_test, dir)
        print('Saved!')

        loss_train = 0
        model.train()

    elif not save_freq_n_epoch:
        print('Saving models and optimizer...')
        save_model_optimizer(model, optimizer, epoch, global_batch_counter_train, global_batch_counter_test, dir)
        print('Saved!')

    if DEBUG:
        break

writer.close()