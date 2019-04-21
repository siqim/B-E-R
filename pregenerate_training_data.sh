dir='./data/datasets/ag_news_csv/'
corpus_name='corpus.pk'
output_dir='lm_data/'
bert_model='bert-base-uncased'

python pregenerate_training_data.py \
--train_corpus $dir$corpus_name \
--output_dir $dir$output_dir \
--bert_model $bert_model \
--do_lower_case \
--epochs_to_generate 3 \
--max_seq_len 64 \
--short_seq_prob 0.1 \
--masked_lm_prob 0.15 \
--max_predictions_per_seq 20