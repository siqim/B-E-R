# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 14:47:52 2019

@author: msq96
"""


import copy
import math
import time
import pickle
import numpy as np

import torch
import torch.nn as nn
from pytorch_pretrained_bert.modeling import BertPreTrainedModel, BertModel, BertForSequenceClassification, BertOnlyMLMHead, BertLayer
from pytorch_pretrained_bert.optimization import BertAdam
from pytorch_pretrained_bert import BertTokenizer
from utils import load_model_optimizer


class BertForMaskedLM(BertPreTrainedModel):

    def __init__(self, config):
        super(BertForMaskedLM, self).__init__(config)

        self.bert = BertModel(config)
        self.cls = BertOnlyMLMHead(config, self.bert.embeddings.word_embeddings.weight)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, cls_embed=None, token_type_ids=None, attention_mask=None, masked_lm_labels=None):

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        embedding_output = self.bert.embeddings(input_ids, token_type_ids) # batch_size x seq_len x hidden_dim

        encoded_layers = self.bert.encoder(embedding_output,
                                              extended_attention_mask,
                                              output_all_encoded_layers=False,
                                              )
        sequence_output = encoded_layers[-1]

        prediction_scores = self.cls(sequence_output)

        return prediction_scores

    @classmethod
    def get(cls, num_labels, dir='./data/datasets/yelp_review_polarity_csv/', model_name='bert-base-uncased'):

        model_weights = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
        param_optimizer = list(model_weights.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
        optimizer = BertAdam(optimizer_grouped_parameters, lr=1e-5)
        model_weights, _, _, _, _ = load_model_optimizer(model_weights, optimizer, dir)

        model = BertForMaskedLM.from_pretrained(pretrained_model_name_or_path=dir+'finetuned_lm/')
        model.bert = model_weights.bert
        model.classifier = model_weights.classifier

        model.cuda()
        model.eval()

        return model


class BertGen(object):

    def __init__(self, model_name, model):

        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = model

        self.CLS = '[CLS]'
        self.SEP = '[SEP]'
        self.MASK = '[MASK]'
        self.mask_id = self.tokenizer.convert_tokens_to_ids([self.MASK])[0]
        self.sep_id = self.tokenizer.convert_tokens_to_ids([self.SEP])[0]
        self.cls_id = self.tokenizer.convert_tokens_to_ids([self.CLS])[0]

    def tokenize_batch(self, batch):
        return [self.tokenizer.convert_tokens_to_ids(sent) for sent in batch]

    def untokenize_batch(self, batch):
        return [self.tokenizer.convert_ids_to_tokens(sent) for sent in batch]

    def detokenize(self, sent):
        """ Roughly detokenizes (mainly undoes wordpiece) """
        new_sent = []
        for i, tok in enumerate(sent):
            if tok.startswith("##"):
                new_sent[len(new_sent) - 1] = new_sent[len(new_sent) - 1] + tok[2:]
            else:
                new_sent.append(tok)
        return new_sent


    def generate_step(self, out, gen_idx, temperature=None, top_k=0, sample=False, return_list=True):
        """ Generate a word from from out[gen_idx]

        args:
            - out (torch.Tensor): tensor of logits of size batch_size x seq_len x vocab_size
            - gen_idx (int): location for which to generate for
            - top_k (int): if >0, only sample from the top k most probable words
            - sample (Bool): if True, sample from full distribution. Overridden by top_k
        """
        logits = out[:, gen_idx]
        if temperature is not None:
            logits = logits / temperature
        if top_k > 0:
            kth_vals, kth_idx = logits.topk(top_k, dim=-1)
            dist = torch.distributions.categorical.Categorical(logits=kth_vals)
            idx = kth_idx.gather(dim=1, index=dist.sample().unsqueeze(-1)).squeeze(-1)
        elif sample:
            dist = torch.distributions.categorical.Categorical(logits=logits)
            idx = dist.sample().squeeze(-1)
        else:
            idx = torch.argmax(logits, dim=-1)
        return idx.tolist() if return_list else idx


    def get_init_text(self, seed_text, max_len, batch_size = 1, rand_init=False):
        """ Get initial sentence by padding seed_text with either masks or random words to max_len """
        batch = [seed_text + [self.MASK] * max_len + [self.SEP] for _ in range(batch_size)]
        #if rand_init:
        #    for ii in range(max_len):
        #        init_idx[seed_len+ii] = np.random.randint(0, len(tokenizer.vocab))

        return self.tokenize_batch(batch)

    def printer(self, sent, should_detokenize=True):
        if should_detokenize:
            sent = self.detokenize(sent)[1:-1]
        print(" ".join(sent))

    '''
    This is the meat of the algorithm. The general idea is

    start from all masks
    repeatedly pick a location, mask the token at that location, and generate from the probability distribution given by BERT
    stop when converged or tired of waiting
    We consider three "modes" of generating:

    generate a single token for a position chosen uniformly at random for a chosen number of time steps
    generate in sequential order (L->R), one token at a time
    generate for all positions at once for a chosen number of time steps
    The generate function wraps and batches these three generation modes. In practice, we find that the first leads to the most fluent samples.
    '''

    def parallel_sequential_generation(self, seed_text, batch_size=10, max_len=15, top_k=0, temperature=None,
                                       max_iter=300, burnin=200, print_every=10, verbose=True):
        """ Generate for one random position at a timestep

        args:
            - burnin: during burn-in period, sample from full distribution; afterwards take argmax
        """
        seed_len = len(seed_text)
        batch = self.get_init_text(seed_text, max_len, batch_size)

        for ii in range(max_iter):
            kk = np.random.randint(0, max_len)
            for jj in range(batch_size):
                batch[jj][seed_len+kk] = self.mask_id
            inp = torch.tensor(batch).cuda()
            out = self.model(inp)
            topk = top_k if (ii >= burnin) else 0
            idxs = self.generate_step(out, gen_idx=seed_len+kk, top_k=topk, temperature=temperature, sample=(ii < burnin))
            for jj in range(batch_size):
                batch[jj][seed_len+kk] = idxs[jj]

            if verbose and np.mod(ii+1, print_every) == 0:
                for_print = self.tokenizer.convert_ids_to_tokens(batch[0])
                for_print = for_print[:seed_len+kk+1] + ['(*)'] + for_print[seed_len+kk+1:]
                print("iter", ii+1, " ".join(for_print))

        return self.untokenize_batch(batch), batch

    def generate(self, n_samples, seed_text="[CLS]", batch_size=10, max_len=25, sample=True,
                 top_k=100, temperature=1.0, burnin=200, max_iter=500, print_every=1):
        # main generation function to call
        sentences = []
        ids_sentences = []
        n_batches = math.ceil(n_samples / batch_size)
        start_time = time.time()
        print('Generating sentences...')
        with torch.no_grad():
            for batch_n in range(n_batches):
                batch, ids_batch = self.parallel_sequential_generation(seed_text, batch_size=batch_size, max_len=max_len, top_k=top_k,
                                                               temperature=temperature, burnin=burnin, max_iter=max_iter,
                                                               verbose=False)

                if (batch_n + 1) % print_every == 0:
                    print("Finished batch %d in %.3fs" % (batch_n + 1, time.time() - start_time))
                    start_time = time.time()

                sentences += batch
                ids_sentences += ids_batch
        return sentences, torch.LongTensor(ids_sentences)


if __name__ == '__main__':
    '''
    Let's call the actual generation function! We'll use the following settings

    max_len (40): length of sequence to generate
    top_k (100): at each step, sample from the top_k most likely words
    temperature (1.0): smoothing parameter for the next word distribution. Higher means more like uniform; lower means more peaky
    burnin (250): for non-sequential generation, for the first burnin steps, sample from the entire next word distribution, instead of top_k
    max_iter (500): number of iterations to run for
    seed_text (["CLS"]): prefix to generate for. We found it crucial to start with the CLS token; you can try adding to it
    '''


    dir = './data/datasets/yelp_review_polarity_csv/'
    model_name = 'bert-base-uncased'
    label_ids, train_loader = pickle.load(open(dir+'train_loader.pk', 'rb'))
    num_labels = len(label_ids)

    model = BertForMaskedLM.get(num_labels=num_labels, dir=dir, model_name=model_name)
    bertgen = BertGen(model_name=model_name, model=model)

    n_samples = 24
    batch_size = 24
    max_len = 40
    top_k = 100
    temperature = 0.7
    burnin = 250
    sample = True
    max_iter = 500


    with torch.no_grad():
        for idx, batch in enumerate(train_loader, 1):
            input_ids, input_mask, label_ids = tuple(t.cuda() for t in batch)
            input_words = np.stack([bertgen.tokenizer.convert_ids_to_tokens(each.cpu().data.numpy().tolist()) for each in input_ids])

            # batch_size x hidden_dim
            _, cls_embed = model.bert(input_ids=input_ids, token_type_ids=None, attention_mask=input_mask, output_all_encoded_layers=False)
            logits = model.classifier(cls_embed)


            # Choose the prefix context
            seed_text = "[CLS]".split()
            bert_sents, ids_bert_sents = bertgen.generate(n_samples, seed_text=seed_text, batch_size=batch_size, max_len=max_len,
                                                          sample=sample, top_k=top_k, temperature=temperature, burnin=burnin, max_iter=max_iter)
            for sent in bert_sents:
              bertgen.printer(sent, should_detokenize=True)

            _, pooled_output = model.bert(input_ids=ids_bert_sents.cuda())
            output_logits = model.classifier(pooled_output)


            break
