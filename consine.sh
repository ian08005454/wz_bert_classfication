CUDA_VISIBLE_DEVICES=0 deepspeed  wz_bert.py --data_path ./data/clause_80k_task.jsonl\
    --batch_size 1\
    --gradient_accumulation_steps 50\
    --num_workers 15 \
    --epoch 2\
    --max_length 512\
    --base_model "hfl/chinese-roberta-wwm-ext-large"\
    --num_class 2\
    --save_steps 1000\
    --eval_steps 1000\
    --training_key question/response\
    --mode train\
    --deepspeed ./ds_no_offload.json\
    --fp16 False\