CUDA_VISIBLE_DEVICES=0 python3  wz_bert.py --data_path ./data/resample_finetune_clause4_512_772.jsonl \
    --training_key content1/content2\
    --batch_size 50 \
    --epoch 10\
    --max_length 512\
    --base_model "hfl/chinese-pert-base"\
    --num_class 2\
    --mode train