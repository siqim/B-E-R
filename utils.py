# -*- coding: utf-8 -*-
"""
Created on Sun Apr 14 17:14:40 2019

@author: msq96
"""


import os
import time
import pandas as pd
import torch
import pickle
import numpy as np
from tqdm import tqdm
from nltk import sent_tokenize
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset


from pytorch_pretrained_bert import BertTokenizer


def normalized_entropy(p):
    m = len(p)
    v = 0
    for each in p:
        v += each * np.log2(each)
    return 1 - (-1/(np.log2(m)) * v)


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


def train(batch, model, loss_fct, optimizer):
    model.zero_grad()

    input_ids, input_mask, label_ids = tuple(t.cuda() for t in batch)

    logits = model(input_ids=input_ids, token_type_ids=None, attention_mask=input_mask, labels=None)
    loss = loss_fct(logits, label_ids)

    loss.backward()
    optimizer.step()

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


class InputFeatures(object):

    def __init__(self, input_ids, input_mask, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.label_id = label_id


class InputExample(object):

    def __init__(self, token_ids, label):
        self.token_ids = token_ids
        self.label = label


def concat_list_of_string(x):
    return ' '.join([each+'.' if each[-1] != '.' else each for each in x])


def build_examples(dir='./data/datasets/ag_news_csv/', filename='train.csv'):
    print('Loading data...')
    data = pd.read_csv(dir+filename, header=None)

    print('Processing data...')
    data.iloc[:, 1:] = data.iloc[:, 1:].fillna('UNK')
    texts = data.iloc[:, 1:].apply(lambda x: concat_list_of_string(x), axis=1)
    labels = data.iloc[:, 0]
    del data

    print('Building examples...')
    examples = [InputExample(text, label) for text, label in zip(texts, labels)]
    label_list = np.unique(labels)
    return examples, label_list



def build_data(dir, tokenizer):
    print('Loading data...')
    train_df = pd.read_csv(dir+'train.csv', header=None)
    test_df = pd.read_csv(dir+'test.csv', header=None)
    train_df['flag'] = 'train'
    test_df['flag'] = 'test'

    print('Concatenating dataframes and text...')
    data = pd.concat([train_df, test_df])
    data.iloc[:, 1:] = data.iloc[:, 1:].fillna('UNK')
    texts = data.iloc[:, 1:-1].apply(lambda x: concat_list_of_string(x), axis=1).values
    labels = data.iloc[:, 0].values
    label_flags = data['flag'].values
    del data

    def parallel_process(doc_text):
        doc_sents = sent_tokenize(doc_text)
        doc_tokenized_sents = [tokenizer.tokenize(each_sent) for each_sent in doc_sents]

        doc_all_tokens = [word_token for tokenized_sent in doc_tokenized_sents for word_token in tokenized_sent]
        doc_all_token_ids = tokenizer.convert_tokens_to_ids(doc_all_tokens, allow_longer=True)
        return doc_tokenized_sents, doc_all_token_ids

    f_corpus = open(dir + 'corpus.pk', 'wb')
    f_train = open(dir + 'train.pk', 'wb')
    f_test = open(dir + 'test.pk', 'wb')

    print('Tokenizing data...')
    for doc_text, doc_label, label_flag in tqdm(zip(texts, labels, label_flags), total=len(labels)):
        doc_tokenized_sents, doc_all_token_ids = parallel_process(doc_text)

        pickle.dump(doc_tokenized_sents, f_corpus)

        if label_flag == 'train':
            pickle.dump(InputExample(doc_all_token_ids, doc_label), f_train)

        elif label_flag == 'test':
            pickle.dump(InputExample(doc_all_token_ids, doc_label), f_test)

    f_corpus.close()
    f_train.close()
    f_test.close()


def pickle_load(fp):

    with open(fp, "rb") as f:
        while 1:
            try:
                yield pickle.load(f)
            except EOFError:
                break


def convert_examples_to_features(dir, fname, max_seq_length, tokenizer):

    examples = pickle_load(dir+fname)
    all_labels = [example.label for example in examples]
    total_num = len(all_labels)

    label_list = np.unique(all_labels)
    label_map = {label : i for i, label in enumerate(label_list)}

    def parallel_process(example):
        token_ids = example.token_ids
        if len(token_ids) > max_seq_length - 2:
            token_ids = token_ids[:(max_seq_length - 2)]

        input_ids = tokenizer.convert_tokens_to_ids(["[CLS]"]) + token_ids + tokenizer.convert_tokens_to_ids(["[SEP]"])
        input_mask = [1] * len(input_ids)

        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding

        label_id = label_map[example.label]

        return InputFeatures(input_ids=input_ids,
                             input_mask=input_mask,
                             label_id=label_id)

    examples = pickle_load(dir+fname)
    features = [parallel_process(example) for example in tqdm(examples, total=total_num)]

    with open(dir+fname.split('.')[0]+'_features.pk', 'wb') as fout:
        pickle.dump(features, fout)

    return features

def save_as_data_loader(dir, flag, batch_size, size_per_class=None):
    features = np.array(pickle.load(open(dir+flag+'_features.pk', 'rb')))

    if size_per_class is not None:
        y = np.array([feature.label_id for feature in features])
        unique_labels = np.unique(y)

        all_idx = []
        for each_label in unique_labels:
            indexes = np.random.choice(np.argwhere(y==each_label).reshape(-1), size_per_class).tolist()
            all_idx.extend(indexes)

        np.random.shuffle(all_idx)

        features = features[all_idx]

        _, counts = np.unique([feature.label_id for feature in features], return_counts=True)

        assert np.all(counts==size_per_class)


    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
    data = TensorDataset(all_input_ids, all_input_mask, all_label_ids)

    if flag == 'train':
        sampler = RandomSampler(data)
    elif flag == 'test':
        sampler = SequentialSampler(data)

    label_ids = np.unique(all_label_ids)
    dataloader = DataLoader(data, sampler=sampler, batch_size=batch_size)

    pickle.dump([label_ids, dataloader], open(dir + '%s_loader.pk'%flag, 'wb'))


if __name__ == '__main__':

    dir = './data/datasets/yelp_review_polarity_csv/'
    model_name = 'bert-base-uncased'

    max_seq_length_train = 64
    max_seq_length_test = 256

    batch_size_train = 24
    batch_size_test = 32

    size_per_class_train = 20
    size_per_class_test = 1000

    tokenizer = BertTokenizer.from_pretrained(model_name)

#    build_data(dir, tokenizer)
#    train_features = convert_examples_to_features(dir=dir, fname='train.pk', max_seq_length=max_seq_length_train, tokenizer=tokenizer)
#    test_features = convert_examples_to_features(dir=dir, fname='test.pk', max_seq_length=max_seq_length_test, tokenizer=tokenizer)

    save_as_data_loader(dir , flag='train', batch_size=batch_size_train, size_per_class=size_per_class_train)
    save_as_data_loader(dir, flag='test', batch_size=batch_size_test, size_per_class=size_per_class_test)
