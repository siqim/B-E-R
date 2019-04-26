python3 finetune_on_pregenerated.py --pregenerated_data ./data/datasets/yelp_review_polarity_csv/lm_data/ --output_dir ./data/datasets/yelp_review_polarity_csv/finetuned_lm/ --bert_model bert-base-uncased --do_lower_case --epochs 3 --train_batch_size 32 --gradient_accumulation_steps 1
