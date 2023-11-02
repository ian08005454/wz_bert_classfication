# %%
import numpy as np
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
import argparse
import transformers
from datasets import load_dataset, load_metric
from transformers import AutoTokenizer
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from datetime import datetime, timezone, timedelta
import logging


# 設定為 +8 時區
tz = timezone(timedelta(hours=+8))
# %%

# %%


def compute_metrics_multi_class(eval_pred):
    pred, labels = eval_pred
    pred = np.argmax(pred, axis=1)
    accuracy = accuracy_score(y_true=labels, y_pred=pred)
    recall = recall_score(y_true=labels, y_pred=pred, average="micro")
    precision = precision_score(y_true=labels, y_pred=pred, average="micro")
    f1 = 2.0*recall*precision/(recall+precision)
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}


def save_log(file_name, log_info):
    '''save in text format'''
    with open(file_name, 'w')as f:
        f.write(log_info)


def tokenizer_data(keys, tokenizer, data, max_length, preprocessing_num_workers=5):
    '''tokenize data'''
    concat_key = keys.split('/')
    if len(concat_key) > 1:
        token_data = data.map(lambda examples:
                              tokenizer(examples[concat_key[0]], examples[concat_key[1]],  max_length=max_length, truncation=True), batched=True,
                              num_proc=preprocessing_num_workers, load_from_cache_file=True, keep_in_memory=False, desc="Running tokenizer on dataset")
    else:
        token_data = data.map(lambda examples:
                              tokenizer(examples[concat_key[0]], max_length=max_length, truncation=True), batched=True,
                              num_proc=preprocessing_num_workers, load_from_cache_file=True, keep_in_memory=False, desc="Running tokenizer on dataset", )
    return token_data


def compute_metrics(eval_pred):
    pred, labels = eval_pred
    pred = np.argmax(pred, axis=1)
    accuracy = accuracy_score(y_true=labels, y_pred=pred)
    recall = recall_score(y_true=labels, y_pred=pred)
    precision = precision_score(y_true=labels, y_pred=pred)
    f1 = f1_score(y_true=labels, y_pred=pred)
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}

# %%


def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    print("loading tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model, use_fast=True)
    if args.mode == "train":
        # training setting
        if args.eval_path:
            print("loading train and eval data")
            train_dataset = load_dataset(
                "json", data_files=args.data_path, split="train")
            val_dataset = load_dataset(
                "json", data_files=args.eval_path, split="train")
        else:
            dataset = load_dataset(
                "json", data_files=args.data_path, split="train")
            shuffled_dataset = dataset.shuffle(seed=42)
            dataset = shuffled_dataset.train_test_split(test_size=0.2)
            train_dataset = dataset["train"]
            val_dataset = dataset["test"]
        # using glue metric for f1 ro acc
        # define tokenizer
        # tokenize data
        print("tokenzing data")
        token_train = tokenizer_data(
            args.training_key, tokenizer, train_dataset, args.max_length, args.num_workers)
        token_val = tokenizer_data(
            args.training_key, tokenizer, val_dataset, args.max_length, args.num_workers)
        # load pretrained model if provided
    else:
        print("loading test data")
        test_dataset = load_dataset(
            "json", data_files=args.data_path)
        token_val = tokenizer_data(
            args.training_key, tokenizer, test_dataset['train'], args.max_length, args.num_workers)
        token_train = token_val
    if args.pretrained_model:
        print("loading prtrained model")
        model = AutoModelForSequenceClassification.from_pretrained(
            args.pretrained_model, num_labels=args.num_class, ignore_mismatched_sizes=False)
    else:
        print("loading base model from huggingface")
        model = AutoModelForSequenceClassification.from_pretrained(
            args.base_model, num_labels=args.num_class)
    # path check
    os.makedirs(args.output_path, exist_ok=True)
    save_log(args.output_path+"/args.txt",str(args))
    # train augment
    resume_from_checkpoint = True if args.overwrite_dir == False else False
    train_args = TrainingArguments(args.output_path+"/train/", evaluation_strategy=args.evaluation_strategy,
                                   save_strategy=args.save_strategy, save_steps=args.save_steps, learning_rate=args.lr,
                                   per_device_train_batch_size=args.batch_size, per_device_eval_batch_size=args.batch_size,
                                   num_train_epochs=args.epoch, weight_decay=args.weight_decay, load_best_model_at_end=True,
                                   metric_for_best_model=args.metric_for_best_model, save_total_limit=args.save_total_limit,
                                   eval_steps=args.eval_steps, deepspeed=args.deepspeed, fp16=args.fp16, 
                                   gradient_accumulation_steps=args.gradient_accumulation_steps,
                                   )
    if args.num_class > 2:
        trainer = Trainer(model, train_args, train_dataset=token_train,
                          eval_dataset=token_val, tokenizer=tokenizer, compute_metrics=compute_metrics_multi_class,
                          )
    else:  # binary
        trainer = Trainer(model, train_args, train_dataset=token_train,
                          eval_dataset=token_val, tokenizer=tokenizer, compute_metrics=compute_metrics
                          )
    if args.mode == "eval":
        print("evaluating...")
        eval_result = trainer.evaluate()
        print(eval_result)
        save_log(args.output_path+"/evaluate1.txt",
                 str(args)+"\n"+str(eval_result))
    else:
        print("training ...")
        trainer.train(resume_from_checkpoint=True)
        model.save_pretrained(args.output_path+"/best_model/")
        print("evaluating...")
        eval_result = trainer.evaluate()
        print(eval_result)
        save_log(args.output_path+"/evaluate.txt",
                 str(args)+"\n"+str(eval_result))
        print("process completed")


# %%
if __name__ == "__main__":
    start_time = datetime.now(tz)
    parser = argparse.ArgumentParser()
    # dataset args
    parser.add_argument('--data_path', default="./dataset.json", type=str,
                        help='dataset path')
    parser.add_argument('--training_key', default="content",
                        type=str, help='training data key in json split with /')
    parser.add_argument('--mode', default="train", type=str,
                        help='train or eval or predict')
    parser.add_argument('--eval_path', type=str,
                        help='eval data path')
    parser.add_argument('--fp16', default=False, type=bool,
                        help='FP16 training')
    parser.add_argument('--train_key', default="train", type=str,
                        help="keyword of training data in json")
    parser.add_argument('--val_key', default="val", type=str,
                        help="keyword of validation data in json")
    parser.add_argument('--predict_key', default="test", type=str,
                        help="keyword of test data in json")
    # training parameter
    parser.add_argument('--gpus', default="0", type=str,)
    parser.add_argument('--base_model', default="bert-base-chinese", type=str,)
    parser.add_argument('--pretrained_model', default="", type=str,)
    parser.add_argument('--max_length', default=512, type=int,)
    parser.add_argument('--num_class', default=2, type=int,)
    parser.add_argument('--lr', default=1e-5, type=float,
                        help='learning rate')
    parser.add_argument('--epoch', default=10, type=int,
                        help='epoch')
    parser.add_argument('--batch_size', default=5, type=int,)
    parser.add_argument('--gradient_accumulation_steps', default=1, type=int,)
    parser.add_argument('--evaluation_strategy', default="steps", type=str,)
    parser.add_argument('--save_strategy', default="steps", type=str,)
    parser.add_argument('--save_steps', default=500, type=int,
                        help="steps for saving")
    parser.add_argument('--eval_steps', default=500, type=int,
                        help="steps for evaluate")
    parser.add_argument('--weight_decay', default=0.01, type=float,)
    parser.add_argument('--metric_for_best_model', default="f1", type=str,)
    parser.add_argument('--save_total_limit', default=3, type=int,)
    parser.add_argument('--deepspeed', default=None, type=str,)
    parser.add_argument('--overwrite_dir',action= "store_false", help="overwrite the output dir")
    # output args
    parser.add_argument('--output_path', default="./output/", type=str,
                        help="save model/predict path")
    parser.add_argument('--local_rank', default=-1, type=int,)
    parser.add_argument('--num_workers', default=5, type=int,
                        help="How many subprocesses to use for data loading. 0 means that the data will be loaded in the main process.")
    args = parser.parse_args()
    print(args)
    main(args)
    print("total time:", datetime.now(tz)-start_time)
