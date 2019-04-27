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


def get_topk(logits, topk, y_pred, test_set):
    probs = torch.softmax(torch.FloatTensor(logits), dim=1)
    y_pred = np.array(y_pred)
    num_classes = probs.shape[1]
    set_len = len(test_set)

    pool = []
    for i in range(num_classes):
        k = np.argsort(-probs[:,i])[: min(topk, set_len//num_classes)]
        pool.extend(k.tolist())

    pool = list(set(pool))

    all_input_ids, all_input_mask, _ = test_set[pool]
    all_label_ids = torch.LongTensor(y_pred[pool])
    print('Acc,', (_==all_label_ids).sum().item() / all_label_ids.shape[0])
    data = TensorDataset(all_input_ids, all_input_mask, all_label_ids)
    return data


def normalized_entropy(p):
    m = len(p)
    v = 0
    for each in p:
        v += each * np.log2(each)
    return 1 - (-1/(np.log2(m)) * v)


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

def save_as_data_loader(dir, flag, size_per_class=None):
    features = np.array(pickle.load(open(dir+flag+'_features.pk', 'rb')))

    if size_per_class is not None:
        y = np.array([feature.label_id for feature in features])
        unique_labels = np.unique(y)

        all_idx = []
        for each_label in unique_labels:
            indexes = np.random.choice(np.argwhere(y==each_label).reshape(-1), size_per_class).tolist()
            all_idx.extend(indexes)

        np.random.shuffle(all_idx)
        used_features = features[all_idx]
        _, counts = np.unique([feature.label_id for feature in used_features], return_counts=True)
        assert np.all(counts==size_per_class)

        if flag == 'train':
            unused_idx = list(set(np.arange(0, y.shape[0])) - set(all_idx))
            unused_features = features[unused_idx]
            assert unused_features.shape[0] == (y.shape[0] - size_per_class * unique_labels.shape[0])

            unused_input_ids = torch.tensor([f.input_ids for f in unused_features], dtype=torch.long)
            unused_input_mask = torch.tensor([f.input_mask for f in unused_features], dtype=torch.long)
            unused_label_ids = torch.tensor([f.label_id for f in unused_features], dtype=torch.long)
            unused_data = TensorDataset(unused_input_ids, unused_input_mask, unused_label_ids)

            pickle.dump([None, unused_data], open(dir + '%s_set_unused.pk'%flag, 'wb'))
    else:
        used_features = features


    all_input_ids = torch.tensor([f.input_ids for f in used_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in used_features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in used_features], dtype=torch.long)
    data = TensorDataset(all_input_ids, all_input_mask, all_label_ids)

    label_ids = np.unique(all_label_ids)

    pickle.dump([label_ids, data], open(dir + '%s_set.pk'%flag, 'wb'))


if __name__ == '__main__':

    dir = './data/datasets/yelp_review_polarity_csv/'
    model_name = 'bert-base-uncased'

    max_seq_length_train = 256
    max_seq_length_test = 256

    size_per_class_train = 20
    size_per_class_test = None

    tokenizer = BertTokenizer.from_pretrained(model_name)

#    build_data(dir, tokenizer)
#    train_features = convert_examples_to_features(dir=dir, fname='train.pk', max_seq_length=max_seq_length_train, tokenizer=tokenizer)
#    test_features = convert_examples_to_features(dir=dir, fname='test.pk', max_seq_length=max_seq_length_test, tokenizer=tokenizer)

    save_as_data_loader(dir , flag='train', size_per_class=size_per_class_train)
    save_as_data_loader(dir, flag='test', size_per_class=size_per_class_test)
