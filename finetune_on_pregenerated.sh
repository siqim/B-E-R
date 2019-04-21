dir='./data/datasets/ag_news_csv/'
input_dir='lm_data/'
output_dir='finetuned_lm/'
bert_model='bert-base-uncased'

python finetune_on_pregenerated.py \
--pregenerated_data $dir$input_dir \
--output_dir $dir$output_dir \
--bert_model $bert_model \
--do_lower_case \
--epochs 3 \
--train_batch_size 32 \
--gradient_accumulation_steps 4