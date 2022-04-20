import argparse
import json
import logging
import os
import random
import sys
import nltk

import pdb

import numpy as np
import torch
import transformers
from transformers import (AutoConfig, AutoModel, BertTokenizer,BertForTokenClassification,
                          DataCollatorForTokenClassification, HfArgumentParser,DataCollatorForSeq2Seq,Seq2SeqTrainer,
                          Seq2SeqTrainingArguments, Trainer, TrainerCallback,AutoModelForSeq2SeqLM)
from transformers.trainer_utils import is_main_process
from datasets import load_metric,Dataset
from utils import DataTrainingArguments, ModelArguments, load_json

import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from modeling_cpt import CPTModel, CPTForConditionalGeneration


parser = argparse.ArgumentParser()
parser.add_argument("--model_path",default='/path/to/model',type=str)
parser.add_argument("--dataset", default="lcsts",type=str)
parser.add_argument("--lr",default=2e-5,type=float)
parser.add_argument("--batch_size",default='10',type=str)
parser.add_argument("--epoch",default='10',type=str)
parser.add_argument("--data_dir",default="/path/to/dataset/",type=str)
args = parser.parse_args()
arg_dict=args.__dict__

logger = logging.getLogger(__name__)

dataset_name=arg_dict['dataset']
outdir_1='output'
if not os.path.exists(outdir_1):
    os.mkdir(outdir_1)

outdir=outdir_1+'/'+dataset_name
if not os.path.exists(outdir):
    os.mkdir(outdir)

seed=len(os.listdir(outdir))+1
outdir=outdir+'/'+str(seed)
length_map={'lcsts':'30','csl':'50','adgen':'128','DuSinc_query':'65'}


args=[
    '--model_name_or_path',arg_dict['model_path'],
    '--do_train','--do_eval','--do_predict',
    '--train_file',os.path.join(arg_dict['data_dir'],'dialogue_topic','train.csv'),
    '--validation_file',os.path.join(arg_dict['data_dir'],'dialogue_topic','dev.csv'),
    '--test_file',os.path.join(arg_dict['data_dir'],'dialogue_topic','test.csv'),
    '--output_dir',outdir,
    '--per_device_train_batch_size',arg_dict['batch_size'],
    '--per_device_eval_batch_size',arg_dict['batch_size'],
    '--overwrite_output_dir',
    '--max_source_length=512',
    '--val_max_target_length='+length_map[arg_dict['dataset']],
    '--predict_with_generate=1',
    '--seed',str(1000*seed),
    '--num_train_epochs',arg_dict['epoch'],
    '--save_strategy','no',
    '--evaluation_strategy','epoch',
    '--learning_rate',str(arg_dict['lr']),
]
parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments))
model_args, data_args, training_args = parser.parse_args_into_dataclasses(args)

"""先不预测"""
training_args.do_predict = False
training_args.fp16 = True


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        
set_seed(training_args.seed)

def load_data(data_path):
    results={'input':[], 'response':[]}
    with open(data_path, encoding='utf-8') as f:
        for line in f.readlines():
            line_lst = line[:-1].split('\t')
            assert(len(line_lst) <= 2)
            results['input'].append(line_lst[0])
            if(len(line_lst) == 2):
                results['response'].append(line_lst[1])
    
        if(len(results['response']) == 0):
            del results['response']
        results=Dataset.from_dict(results)
    return results

datasets={}
data_files = {}
if data_args.train_file is not None:
    data_files["train"] = data_args.train_file
if data_args.validation_file is not None:
    data_files["validation"] = data_args.validation_file
if data_args.test_file is not None:
    data_files["test"] = data_args.test_file
for key in data_files:
    print(key)
    datasets[key]=load_data(data_files[key])

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger.setLevel(logging.INFO if is_main_process(training_args.local_rank) else logging.WARN)

# Log on each process the small dialogue:
logger.warning(
    f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
    + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
)
# Set the verbosity to info of the Transformers logger (on main process only):
if is_main_process(training_args.local_rank):
    transformers.utils.logging.set_verbosity_info()
logger.info("Training/evaluation parameters %s", training_args)

tokenizer=BertTokenizer.from_pretrained(model_args.model_name_or_path)
model=CPTForConditionalGeneration.from_pretrained(model_args.model_name_or_path)
model.config.max_length=data_args.val_max_target_length

text_column='input'
summary_column='response'
column_names = datasets["train"].column_names
column_names_test = datasets['test'].column_names
max_target_length = data_args.val_max_target_length
padding=False

def preprocess_function(examples):
    inputs = examples[text_column]
    targets = examples[summary_column]
    model_inputs = tokenizer(inputs, max_length=data_args.max_source_length, padding=padding, truncation=True)

    # Setup the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=max_target_length, padding=padding, truncation=True)


    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def preprocess_function_test(examples):
    inputs = examples[text_column]
    model_inputs = tokenizer(inputs, max_length=data_args.max_source_length, padding=padding, truncation=True)
    
    return model_inputs


if training_args.do_train:
    train_dataset = datasets["train"]
    if "train" not in datasets:
        raise ValueError("--do_train requires a train dataset")
    if data_args.max_train_samples is not None:
        train_dataset = train_dataset.select(range(data_args.max_train_samples))
    train_dataset = train_dataset.map(
        preprocess_function,
        batched=True,
        num_proc=data_args.preprocessing_num_workers,
        remove_columns=column_names,
        load_from_cache_file=not data_args.overwrite_cache,
    )

if training_args.do_eval:
    max_target_length = data_args.val_max_target_length
    if "validation" not in datasets:
        raise ValueError("--do_eval requires a validation dataset")
    eval_dataset = datasets["validation"]
    if data_args.max_val_samples is not None:
        eval_dataset = eval_dataset.select(range(data_args.max_val_samples))
    eval_dataset = eval_dataset.map(
        preprocess_function,
        batched=True,
        num_proc=data_args.preprocessing_num_workers,
        remove_columns=column_names,
        load_from_cache_file=not data_args.overwrite_cache,
    )

if training_args.do_predict:
    max_target_length = data_args.val_max_target_length
    if "test" not in datasets:
        raise ValueError("--do_predict requires a test dataset")
    test_dataset = datasets["test"]
    if data_args.max_test_samples is not None:
        test_dataset = test_dataset.select(range(data_args.max_test_samples))
    test_dataset = test_dataset.map(
        preprocess_function_test,
        batched=True,
        num_proc=data_args.preprocessing_num_workers,
        remove_columns=column_names,
        load_from_cache_file=not data_args.overwrite_cache,
    )


max_eval_num=30000
if len(eval_dataset)>max_eval_num:
    eval_dataset=Dataset.from_dict(eval_dataset[:max_eval_num])
print(len(eval_dataset))


# Data collator
label_pad_token_id = -100 if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
data_collator = DataCollatorForSeq2Seq(
    tokenizer,
    model=model,
    label_pad_token_id=label_pad_token_id,
    pad_to_multiple_of=8 if training_args.fp16 else None,
)



# Metric
from rouge import Rouge 
rouge = Rouge()

def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]

    # # rougeLSum expects newline after each sentence
    # preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
    # labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

    while '' in preds:
        idx=preds.index('')
        preds[idx]='。'

    return preds, labels

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    if data_args.ignore_pad_token_for_loss:
        # Replace -100 in the labels as we can't decode them.
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
   
    # Some simple post-processing
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
    scores = rouge.get_scores(decoded_preds, decoded_labels,avg=True)
    for key in scores:
        scores[key]=scores[key]['f']*100

    result=scores

    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items()}
    return result

class TestCallback(TrainerCallback):
    def on_evaluate(self, args, state, control, **kwargs):
        predictions, labels, metrics = trainer.predict(test_dataset, metric_key_prefix="predict")
        metrics['epoch']=state.epoch
        state.log_history.append(metrics)

class ValidCallback(TrainerCallback):
    def on_evaluate(self, args, state, control, **kwargs):
        predictions, labels, metrics = trainer.predict(eval_dataset, metric_key_prefix="predict")
        metrics['epoch']=state.epoch
        state.log_history.append(metrics)

# Initialize our Trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset if training_args.do_train else None,
    eval_dataset=eval_dataset if training_args.do_eval else None,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics if training_args.predict_with_generate else None,
    callbacks=[ValidCallback],
)

# Training
if training_args.do_train:
    train_result = trainer.train()
    trainer.save_model()  # Saves the tokenizer too for easy upload

    metrics = train_result.metrics
    max_train_samples = (
        data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
    )
    metrics["train_samples"] = min(max_train_samples, len(train_dataset))

    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()


outdir = 'output/DuSinc_query/valid_result'
querys = []
cnt = 0
for src in datasets['validation']['input']:
    input_ids = tokenizer.encode(src, return_tensors='pt')
    if(input_ids.shape[1] > 512):
        trunc = input_ids.shape[1] - 512 + 10
        # 从后面截断context
        src_lst = src.split('[SEP]')
        for j in range(1, len(src_lst)):
            if(len(src_lst[j]) <= trunc):
                trunc -= len(src_lst[j])
                src_lst[j] = ''
            else:
                src_lst[j] = src_lst[j][trunc:]
                trunc = 0
                break
        src_lst_new = []
        for j in src_lst:
            if j != '':
                src_lst_new.append(j)
        src_new = '[SEP]'.join(src_lst_new)
        input_ids = tokenizer.encode(src_new, return_tensors='pt')
    logits = model.generate(input_ids.cuda(), num_beams=4)
    tgt = tokenizer.decode(logits.squeeze())
    querys.append(tgt.replace("[SEP]", "").replace(
        "[CLS]", "").replace(" ", ""))
    print("decode sentence num : {}".format(cnt))
    cnt += 1
output_val_preds_file = os.path.join(
    outdir, "valid_generations.txt")
with open(output_val_preds_file, "w", encoding='UTF-8') as writer:
    writer.write("\n".join(querys))

output_valid_file = os.path.join(
    outdir, "valid.txt")
valid_sentences = []
for q in datasets['validation']['response']:
    valid_sentences.append(q.replace(" ", ""))
with open(output_valid_file, "w", encoding='UTF-8') as writer:
    writer.write("\n".join(datasets['validation']['response']))

"""
if trainer.is_world_process_zero():
    if training_args.predict_with_generate:
        predictions, labels, metrics = trainer.predict(test_dataset, metric_key_prefix="predict")
        test_preds = tokenizer.batch_decode(
            predictions, skip_special_tokens=True,
        )
        test_preds = [pred.strip() for pred in test_preds]
        output_test_preds_file = os.path.join(training_args.output_dir, "test_generations.txt")
        with open(output_test_preds_file, "w",encoding='UTF-8') as writer:
            writer.write("\n".join(test_preds))
"""

if not os.path.exists(outdir):
    os.mkdir(outdir)
querys = []
for src in datasets['test']['input']:
    input_ids = tokenizer.encode(src, return_tensors = 'pt')
    logits = model.generate(input_ids.cuda(),num_beams=4)
    tgt = tokenizer.decode(logits.squeeze())
    querys.append(tgt)
output_test_preds_file = os.path.join(training_args.output_dir, "test_generations.txt")
with open(output_test_preds_file, "w",encoding='UTF-8') as writer:
    writer.write("\n".join(querys))
