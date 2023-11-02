pred_folder="0522_clu_w_noise_npid_folds"
pred_data="all_test.json"
python3 bert_main.py --data_path ./pert_large/claim_data/$pred_data\
    --gpus 0\
    --batch_size 8\
    --output_path pert_large/claim/$pred_folder/\
    --max_length 512\
    --base_model "hfl/chinese-pert-large"\
    --predict_only True\
    --prediction_name "${pred_key}_f0"\
    --predict_key test\
    --num_class 2\
    --mode claim 