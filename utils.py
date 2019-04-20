# -*- coding: utf-8 -*-
"""
Created on Sun Apr 14 17:14:40 2019

@author: msq96
"""


import os
import pandas as pd
import torch
import pickle
import numpy as np
from tqdm import tqdm
from nltk import word_tokenize
from joblib import Parallel, delayed


def build_embed_files(fname='./data/embeddings/wiki-news-300d-1M-subword.vec'):
    vocab_size = 300000
    embed_dim = 300

    embed_matrix = torch.zeros((vocab_size, embed_dim), dtype=torch.float32)
    word2idx = {}
    idx2word = {}

    idx2word[0] = 'zero_pad'
    word2idx['zero_pad'] = 0

    idx2word[1] = 'septor'
    word2idx['septor'] = 1


    fin = open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())

    for idx in tqdm(range(2, vocab_size)):
        line = fin.readline()
        tokens = line.rstrip().split(' ')
        word = tokens[0]
        embed = torch.FloatTensor(list(map(float, tokens[1:])))

        embed_matrix[idx] = embed
        word2idx[word] = idx
        idx2word[idx] = word

    pickle.dump(idx2word, open('./data/embeddings/idx2word.pk', 'wb'))
    pickle.dump(word2idx, open('./data/embeddings/word2idx.pk', 'wb'))
    pickle.dump(embed_matrix, open('./data/embeddings/embed_matrix.pk', 'wb'))


def pad_seq(seq, max_seq, pad_word='zero_pad'):
    seq_len = len(seq)
    if seq_len >= max_seq:
        return seq[: max_seq]
    else:
        num_pad = max_seq - seq_len
        return seq + [pad_word]*num_pad


def build_data(dir='./data/datasets/amazon_review_full_csv/', filename='train.csv',
               max_seq=200, max_seq_percentile=95, sep=' septor ', pad='zero_pad'):
    print('Loading data...')
    data = pd.read_csv(dir+filename, header=None)
    data.iloc[:, 1:] = data.iloc[:, 1:].fillna('UNK')
    data = data.values

    y = torch.LongTensor(data[:, 0].astype(np.int64))

    print('Tokenizing sentences...')
    if data.shape[1] == 3:
        data = data[:, [1, 2]]
        X = Parallel(n_jobs=8)(delayed(word_tokenize)(each[0] + sep + each[1]) for each in tqdm(data))
        del data
    elif data.shape[1] == 2:
        data = data[:, 1]
        assert data.shape == (data.shape[0],)
        X = Parallel(n_jobs=8)(delayed(word_tokenize)(each) for each in tqdm(data))
        del data


    print('Turn words to indexes...')
    word2idx = pickle.load(open('./data/embeddings/word2idx.pk', 'rb'))
    if max_seq is None:
        max_seq = int(np.percentile([len(sent) for sent in X], max_seq_percentile))

    X = torch.LongTensor([[word2idx.get(word, word2idx['UNK']) for word in pad_seq(seq, max_seq=max_seq, pad_word=pad)] \
                           for seq in tqdm(X)])

    print('Saving data...')
    with open(dir+filename.split('.')[0]+'.pk', 'wb') as fout:
        pickle.dump([X, y], fout, protocol=4)
    return X, y, max_seq


def save_model_optimizer(model, optimizer, epoch, global_batch_counter_train, global_batch_counter_test, scheduler, dir):
   state = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),

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

def load_model_optimizer(model, optimizer, scheduler, dir):
    DIR_MODEL = dir + 'models/params/'
    params_list = os.listdir(DIR_MODEL)
    if params_list:
        print('Loading lastest checkpoint...')
        states = load_lastest_states(DIR_MODEL, params_list)

        model.load_state_dict(states['model'])
        optimizer.load_state_dict(states['optimizer'])
        scheduler.load_state_dict(states['scheduler'])

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

if __name__ == '__main__':
    _, _, max_seq = build_data(dir='./data/datasets/amazon_review_full_csv/', filename='train.csv',
                               max_seq=None, max_seq_percentile=95, sep=' septor ', pad='zero_pad')
    build_data(dir='./data/datasets/amazon_review_full_csv/', filename='test.csv',
               max_seq=max_seq, sep=' septor ', pad='zero_pad')



