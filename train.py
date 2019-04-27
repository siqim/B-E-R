# -*- coding: utf-8 -*-
"""
Created on Sun Apr 14 17:11:55 2019

@author: msq96
"""


import pickle
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, ConcatDataset
from tensorboardX import SummaryWriter
from pytorch_pretrained_bert.modeling import BertForSequenceClassification
from pytorch_pretrained_bert.optimization import BertAdam
from sklearn.metrics import f1_score

from utils import train, test, save_model_optimizer, load_model_optimizer, try_mkdir, get_topk

bootstrapping_start_epoch = 4
bootstrapping_increase_coef = 1.5
bootstrapping_max_usage = 0.75
gradient_accumulation_steps = 4
LOG_NAME = 'finetune-all'
DEBUG = False
EPOCH = 3
save_freq_per_epoch = 0
save_freq_n_epoch = 1
assert bool(save_freq_n_epoch) != bool(save_freq_per_epoch)

dir = './data/datasets/yelp_review_polarity_csv/'

batch_size_train = 32
batch_size_test = 32
try_mkdir(dir+'models')
try_mkdir(dir+'models/logs')
try_mkdir(dir+'models/params')


print('Building data...')
label_ids, raw_train_set = pickle.load(open(dir+'train_set.pk', 'rb'))
_, test_set = pickle.load(open(dir+'test_set.pk', 'rb'))

_, train_set_unused = pickle.load(open(dir+'train_set_unused.pk', 'rb'))

batch_size_train = batch_size_train // gradient_accumulation_steps
train_loader = DataLoader(raw_train_set, sampler=RandomSampler(raw_train_set), batch_size=batch_size_train)
test_loader = DataLoader(test_set, sampler=SequentialSampler(test_set), batch_size=batch_size_test)


num_labels = len(label_ids)
num_train_batch = len(train_loader)
test_per_n_batch_one_epoch = num_train_batch // save_freq_per_epoch if save_freq_per_epoch!=0 else 0
#q = ((int(len(raw_train_set)//num_labels * bootstrapping_increase_coef)) * num_labels + len(raw_train_set)) / len(raw_train_set) / save_freq_n_epoch
#n = np.log(len(test_set)*bootstrapping_max_usage / len(raw_train_set)) / np.log(q) + 1
#num_train_optimization_steps = int(len(raw_train_set)*(1-q**n)/(1-q) / batch_size_train  / gradient_accumulation_steps)
num_train_optimization_steps = EPOCH*num_train_batch / gradient_accumulation_steps
topk_pre_class = int(len(raw_train_set)//num_labels * bootstrapping_increase_coef)


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
#optimizer = torch.optim.Adam(optimizer_grouped_parameters, lr=1e-5)

print('Loading models and optimizer...')
model, optimizer, current_epoch, global_batch_counter_train, global_batch_counter_test = load_model_optimizer(model, optimizer, dir)


print('Start training!')
writer = SummaryWriter(dir+'models/logs/%s/'%LOG_NAME)

loss_fct = torch.nn.CrossEntropyLoss()
model.train()
for epoch in range(current_epoch, EPOCH):
    loss_train = 0

    for idx, batch in enumerate(tqdm(train_loader), 1):

        loss, correct = train(batch, model, loss_fct, optimizer, idx, gradient_accumulation_steps)

        writer.add_scalar('loss_train_batch/loss_train_batch', loss, global_batch_counter_train)
        writer.add_scalar('accuracy_train_batch/accuracy_train_batch', correct/batch[0].shape[0], global_batch_counter_train)
        writer.add_scalar('epoch/lr', optimizer.get_lr()[-1], global_batch_counter_train)

        global_batch_counter_train += 1
        loss_train += loss


        if (test_per_n_batch_one_epoch and (idx % test_per_n_batch_one_epoch == 0)) or (DEBUG and idx == 4 and test_per_n_batch_one_epoch):
            print('Start testing...')
            loss_test = 0
            total = 0
            correct = 0
            y_pred = []
            y_true = []
            model.eval()
            with torch.no_grad():
                for test_idx, batch in enumerate(tqdm(test_loader), 1):

                    loss, correct, total, y_pred_batch, y_true_batch, _ = test(batch, model, loss_fct, correct, total)
                    y_pred.extend(y_pred_batch)
                    y_true.extend(y_true_batch)

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

    if ((save_freq_n_epoch and ((epoch+1) % save_freq_n_epoch == 0)) and (bootstrapping_start_epoch and epoch+1 >= bootstrapping_start_epoch)) \
        or (not bootstrapping_start_epoch and (save_freq_n_epoch and ((epoch+1) % save_freq_n_epoch == 0))):

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

        if bootstrapping_start_epoch and topk_pre_class*num_labels <= len(test_set) * bootstrapping_max_usage:
            print('Bootstrapping...')
            extra_train_set = get_topk(logits, topk_pre_class, y_pred, test_set)
            train_set = ConcatDataset([raw_train_set, extra_train_set])
            train_loader = DataLoader(train_set, sampler=RandomSampler(train_set), batch_size=batch_size_train)
            topk_pre_class = int(len(train_set)//num_labels * bootstrapping_increase_coef)

        loss_train = 0
        model.train()

    elif not save_freq_n_epoch:
        print('Saving models and optimizer...')
        save_model_optimizer(model, optimizer, epoch, global_batch_counter_train, global_batch_counter_test, dir)
        print('Saved!')

    if DEBUG:
        break

writer.close()







class Train(self):

    def __init__(self, dir, train_set, test_set, batch_size_train, batch_size_test, log_name):


    def save_model_optimizer(model, optimizer, epoch, global_batch_counter_train, global_batch_counter_test, dir):
       state = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),

                'epoch': epoch,
                'global_batch_counter_train': global_batch_counter_train,
                'global_batch_counter_test': global_batch_counter_test
                }

       torch.save(state, open(dir+'models/params/MODEL_%d.tar'%global_batch_counter_train, 'wb'))

    def load_lastest_states(params_dir, params_list):
        lastest_states_idx = np.argmax([int(each_params.split('_')[1][:-4]) for each_params in params_list])
        lastest_states_path = params_dir + params_list[lastest_states_idx]
        lastest_states = torch.load(open(lastest_states_path, 'rb'))
        return lastest_states

    def load_model_optimizer(model, optimizer, dir):
        DIR_MODEL = dir + 'models/params/'
        params_list = os.listdir(DIR_MODEL)
        if params_list:
            print('Loading lastest checkpoint...')
            states = load_lastest_states(DIR_MODEL, params_list)

            model.load_state_dict(states['model'])
            optimizer.load_state_dict(states['optimizer'])

            current_epoch = states['epoch'] + 1
            global_batch_counter_train = states['global_batch_counter_train']
            global_batch_counter_test = states['global_batch_counter_test']

            return model, optimizer, current_epoch, global_batch_counter_train, global_batch_counter_test

        else:
            return model, optimizer, 0, 0, 0


    def try_mkdir(path):
        if not os.path.isdir(path):
            os.mkdir(path)
            return True
        else:
            return False


    def train(batch, model, loss_fct, optimizer, idx, gradient_accumulation_steps):

        input_ids, input_mask, label_ids = tuple(t.cuda() for t in batch)

        logits = model(input_ids=input_ids, token_type_ids=None, attention_mask=input_mask, labels=None)
        loss = loss_fct(logits, label_ids)
        loss = loss / gradient_accumulation_steps
        loss.backward()


        if idx % gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        _, predicted = torch.max(logits.data, 1)
        correct = (predicted == label_ids).sum().item()

        return loss.item(), correct

    def test(batch, model, loss_fct, correct, total):

        input_ids, input_mask, label_ids = tuple(t.cuda() for t in batch)

        logits = model(input_ids=input_ids, token_type_ids=None, attention_mask=input_mask, labels=None)
        loss = loss_fct(logits, label_ids)

        _, predicted = torch.max(logits.data, 1)
        correct += (predicted == label_ids).sum().item()
        total += label_ids.size(0)

        return loss.item(), correct, total, predicted.cpu().data.numpy().tolist(), label_ids.cpu().data.numpy().tolist(), logits.cpu().data.numpy().tolist()











