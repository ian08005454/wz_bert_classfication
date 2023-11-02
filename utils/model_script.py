from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel
import argparse
import os
import torch
import numpy as np
# useage:
# python3 evl.py -m './output/distiluse-base-multilingual-cased-v1_772_pair_sent_2023-08-03-02-17' --mode vecter


def pre_process(test_sent, sep, batched):
    if batched:
        for i in range(len(test_sent)):
            if type(test_sent[i]) == list:
                test_sent[i] = test_sent[i][0] + sep + test_sent[i][1]
    else:
        if type(test_sent) == list:
            test_sent = test_sent[0] + sep + test_sent[1]
    return test_sent


def get_output(base_model, model_path, test_sent, mode, batched):
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if mode == 'class':
        model = AutoModelForSequenceClassification.from_pretrained(
            model_path).half().cuda()
    else:
        model = AutoModel.from_pretrained(model_path).half().cuda()
    test_sent = pre_process(test_sent, tokenizer.sep_token, batched)
    input_id = tokenizer(test_sent, return_tensors="pt",
                         padding=True, truncation=True, max_length=512, add_special_tokens=True).to('cuda')
    predictions = model(**input_id)
    if mode == 'class':
        probs = np.argmax(predictions.logits.detach().cpu().numpy(), axis=1)
        return probs
    else:
        vecter = []
        probs = predictions[0].detach().cpu().numpy()
        for prob in probs:
            vecter.append(prob[-1])
        return vecter


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", '-m', type=str, help="model path")
    parser.add_argument("--test_sent", default=[
                        "現聲請人對相對人提出本件聲請給付扶養費事件，聲請人係與受監護人之利益相反或依法不得代理，爰聲請法院應依法為相對人選任本件事件之特別代理人。", "聲請人還有謀生能力，自己還是可以工作賺錢等語。"])
    parser.add_argument("--mode", default="class", type=str, )
    parser.add_argument(
        "--base_model", default="sentence-transformers/distiluse-base-multilingual-cased-v1", type=str, )
    parser.add_argument("--batched", default=False,
                        type=bool, help="batch input or not")
    args = parser.parse_args()
    pred = get_output(args.base_model, os.path.join(
        args.model_path, 'best_model'), args.test_sent, args.mode, args.batched)
    print(pred)
