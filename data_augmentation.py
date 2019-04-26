# -*- coding: utf-8 -*-
"""
Created on Sun Apr 21 14:49:27 2019

@author: msq96
"""


import torch
import pickle
from pytorch_pretrained_bert.modeling import BertPreTrainedModel, BertModel, BertOnlyMLMHead, BertEmbeddings, \
                                             BertPooler, BertEncoder, BertForSequenceClassification
from torch.nn import CrossEntropyLoss
from pytorch_pretrained_bert.optimization import BertAdam
from utils import load_model_optimizer
from pytorch_pretrained_bert import BertTokenizer
import numpy as np
from BertGen import BertForMaskedLM, BertGen




if __name__ == '__main__':
    dir = './data/datasets/yelp_review_polarity_csv/'
    model_name = 'bert-base-uncased'
    label_ids, train_loader = pickle.load(open(dir+'train_loader.pk', 'rb'))
    num_labels = len(label_ids)


    model = BertForMaskedLM.get(num_labels=num_labels, dir=dir, model_name=model_name)

    nltk = BertGen(model_name=model_name, model=model)





    with torch.no_grad():
        for idx, batch in enumerate(train_loader, 1):
            input_ids, input_mask, label_ids = tuple(t.cuda() for t in batch)
            input_words = np.stack([tokenizer.convert_ids_to_tokens(each.cpu().data.numpy().tolist()) for each in input_ids])

            # batch_size x hidden_dim
            _, pooled_output = model.bert(input_ids=input_ids, token_type_ids=None, attention_mask=input_mask, output_all_encoded_layers=False)
            n = torch.distributions.Normal(0, 0.01)
            pooled_output += n.sample((24,768)).cuda()
            logits = model.classifier(pooled_output)


            token_type_ids = torch.zeros_like(input_ids)
            masked_input_ids = input_ids.clone()

        with torch.no_grad():
            for masked_index in range(1, input_ids.shape[1]-1):
#                mask_start = masked_index
#                mask_end = masked_index + 2
#                mask = [mask_start, mask_end]

                masked_input_ids[:, masked_index] = mask_id

                embedding_output = model.bert.embeddings(masked_input_ids, token_type_ids) # batch_size x seq_len x hidden_dim
                embedding_output[:, 0, :] = pooled_output

                predictions = model.forward(embedding_output, attention_mask=input_mask)
                predicted_index = torch.argmax(predictions[:, masked_index], dim=1)

                masked_input_ids[:, masked_index] = predicted_index
