from lib2to3.pgen2.tokenize import tokenize
import os
import sys
import random
from tqdm.auto import tqdm

import argparse
import json
import logging
import datasets
from datasets import Dataset, load_metric
import pdb

import math
import numpy as np
import torch
import transformers
from transformers import (AdamW, AutoConfig, BertTokenizer, HfArgumentParser,DataCollatorForSeq2Seq,Seq2SeqTrainer,
                          Seq2SeqTrainingArguments, TrainerCallback, AutoModelForSeq2SeqLM, get_scheduler, SchedulerType)
from transformers.trainer_utils import is_main_process

from accelerate import Accelerator,DistributedDataParallelKwargs
from accelerate.utils import set_seed
from torch.utils.data import DataLoader

from cal_metric import compute_BLEU_batch, compute_f1_batch, compute_distinct_batch

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from modeling_cpt import CPTForConditionalGeneration

import wandb
import socket


logger = logging.getLogger(__name__)

#os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

def parse_args():
    '''
    Description: 
    Arguments: 
    Returns: 
    Others: 
    '''
    parser = argparse.ArgumentParser(description="Finetune a mengzi-t5-base model")
    # 数据相关
    parser.add_argument("--model_path", type=str, default="Langboat/mengzi-t5-base")
    parser.add_argument("--data_dir", type=str, default="data/DuSinc_dial")
    parser.add_argument("--output_dir", type=str, default="output/example2")
    parser.add_argument("--train_file", type=str, default="data/DuSinc_dial/train.csv")
    parser.add_argument("--validation_file", type=str, default="data/DuSinc_dial/dev.csv")
    parser.add_argument("--test_file", type=str, default="data/DuSinc_dial/test.csv")
    parser.add_argument("--max_source_length", type=int, default=512)
    parser.add_argument("--max_target_length", type=int, default=256)  # 任务不同长度不同
    parser.add_argument("--mode", type=str, default="train", choices=["train", "validation", "test"])  # 运行模式
    # 参数相关
    parser.add_argument("--per_device_batch_size", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--num_train_epochs", type=int, default=2)
    parser.add_argument("--max_train_steps", type=int, default=None, help="Total number of training steps to perform. If provided, overrides num_train_epochs.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--lr_scheduler_type", type=SchedulerType, default="linear", choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"])
    parser.add_argument("--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_beams", type=int, default=4)
    parser.add_argument("--checkpointing_steps", type=str, default="epoch", help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.")
    parser.add_argument("--context_pos", type=int, default=2)
    # Wandb
    parser.add_argument("--team_name", type=str, default="ruc-bupt")
    parser.add_argument("--project_name", type=str, default="Paddle SINC")
    parser.add_argument("--experiment_name", type=str)
    parser.add_argument("--scenario_name", type=str)
    
    args = parser.parse_args()
    return args

def load_data(data_path):
    results={'input':[], 'query':[]}
    with open(data_path, encoding='utf-8') as f:
        for line in f.readlines():
            line_lst = line[:-1].split('\t')
            assert(len(line_lst) <= 2)
            results['input'].append(line_lst[0])
            if(len(line_lst) == 2):
                results['query'].append(line_lst[1])
    
        if(len(results['query']) == 0):
            del results['query']
        results=Dataset.from_dict(results)
    return results

def load_dataset(args):
    datasets={}
    data_files = {}
    if args.train_file is not None:
        data_files["train"] = args.train_file
    if args.validation_file is not None:
        data_files["validation"] = args.validation_file
    if args.test_file is not None:
        data_files["test"] = args.test_file
    for key in data_files:
        datasets[key]=load_data(data_files[key])
    return datasets


def main():

    args = parse_args()
    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example
    #ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    #accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])
    accelerator = Accelerator()
    
    # repo creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    wandb.init(config=args,
               project=args.project_name,
               entity=args.team_name,
               notes=socket.gethostname(),
               name=args.experiment_name+" seed:"+str(args.seed),
               group=args.scenario_name,
               dir=str(args.output_dir),
               job_type="training",
               reinit=True)
    
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state)
    # Setup logging, we only want one process per machine to log things on the screen.
    # accelerator.is_local_main_process is only True for one process per machine.
    logger.setLevel(logging.INFO if accelerator.is_local_main_process else logging.ERROR)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # seed
    if args.seed is not None:
        set_seed(args.seed)

    raw_datasets = load_dataset(args)

    # load model
    config = AutoConfig.from_pretrained(args.model_path)
    #config.num_hidden_layers = 10
    tokenizer=BertTokenizer.from_pretrained(args.model_path)
    model=CPTForConditionalGeneration.from_pretrained(args.model_path, config=config)
    model.config.max_length=args.max_target_length

    def preprocess_function_not_test(examples):
        """考虑加一个手动truncation，把context从前面截断"""
        inputs = examples['input']
        targets = examples['query']

        max_len = args.max_source_length
        inputs_trunc = []
        for i in inputs:
            input_encode = tokenizer.encode(i)
            if(len(input_encode) > max_len):
                input_trunc = []
                sep_cnt = 0
                trunc_len = len(input_encode) - max_len
                for token in input_encode:
                    if(token == tokenizer.convert_tokens_to_ids('[SEP]')):
                        sep_cnt += 1
                    if(sep_cnt < args.context_pos):
                        input_trunc.append(token)
                        continue
                    if(sep_cnt == args.context_pos and token == tokenizer.convert_tokens_to_ids('[SEP]')):
                        input_trunc.append(token)
                        continue
                    # context位置
                    if(trunc_len > 0):
                        trunc_len -= 1
                        continue
                    input_trunc.append(token)
                assert(len(input_trunc) <= max_len)
                input_encode = input_trunc
            inputs_trunc.append(tokenizer.decode(input_encode[1:-1]).replace(" ", ""))
        inputs = inputs_trunc
        model_inputs = tokenizer(inputs, max_length=args.max_source_length, padding=False, truncation=True)

        # setup the tokenizer for targets
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(targets, max_length=args.max_target_length, padding=False, truncation=True)

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs


    def preprocess_function_test(examples):
        """手动加一个truncation"""
        inputs = examples['input']

        max_len = args.max_source_length
        inputs_trunc = []
        for i in inputs:
            input_encode = tokenizer.encode(i)
            if(len(input_encode) > max_len):
                input_trunc = []
                sep_cnt = 0
                trunc_len = len(input_encode) - max_len
                for token in input_encode:
                    if(token == tokenizer.convert_tokens_to_ids('[SEP]')):
                        sep_cnt += 1
                    if(sep_cnt < args.context_pos):
                        input_trunc.append(token)
                        continue
                    if(sep_cnt == args.context_pos and token == tokenizer.convert_tokens_to_ids('[SEP]')):
                        input_trunc.append(token)
                        continue
                    # context位置
                    if(trunc_len > 0):
                        trunc_len -= 1
                        continue
                    input_trunc.append(token)
                assert(len(input_trunc) <= max_len)
                input_encode = input_trunc
            inputs_trunc.append(tokenizer.decode(input_encode[1:-1]).replace(" ", ""))
        inputs = inputs_trunc
        model_inputs = tokenizer(inputs, max_length=args.max_source_length, padding=False, truncation=True)
        return model_inputs

    def postprocess_text(preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [[label.strip()] for label in labels]

        """while '' in preds:
            idx = preds.index('')
            preds[idx] = '。'"""

        return preds, labels
        
    with accelerator.main_process_first():
        if args.mode == "train":
            train_dataset = raw_datasets["train"]
            train_dataset = train_dataset.map(
                preprocess_function_not_test, 
                batched=True,
                remove_columns=raw_datasets["train"].column_names,
            )
            eval_dataset = raw_datasets["validation"]
            eval_dataset = eval_dataset.map(
                preprocess_function_not_test, 
                batched=True,
                remove_columns=raw_datasets["validation"].column_names,
            )
            test_dataset = raw_datasets["test"]
            test_dataset = test_dataset.map(  # 注意test有区别
                preprocess_function_test,
                batched=True,
                remove_columns=raw_datasets["test"].column_names,
            )
            # CPT doesn't use token_type_ids
            train_dataset = train_dataset.remove_columns('token_type_ids')
            eval_dataset = eval_dataset.remove_columns('token_type_ids')
            test_dataset = test_dataset.remove_columns('token_type_ids')
    
    # data collator
    label_pad_token_id = tokenizer.pad_token_id
    data_collator = DataCollatorForSeq2Seq(
        tokenizer, 
        model=model, 
        label_pad_token_id=label_pad_token_id, 
        pad_to_multiple_of=8 if accelerator.use_fp16 else None,
    )

    if args.mode == "test":
        test_dataloader = DataLoader(test_dataset, shuffle=True, collate_fn=data_collator, batch_size=args.per_device_batch_size)
    if args.mode == "train":
        train_dataloader = DataLoader(train_dataset, shuffle=True, collate_fn=data_collator, batch_size=args.per_device_batch_size)
        eval_dataloader = DataLoader(eval_dataset, shuffle=True, collate_fn=data_collator, batch_size=args.per_device_batch_size)
        # optimizer
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": args.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

        # scheduler and math
        num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps / accelerator.num_processes)
        if args.max_train_steps is None:
            args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        else:
            args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

        lr_scheduler = get_scheduler(
            name=args.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=args.num_warmup_steps,
            num_training_steps=args.max_train_steps,
        )

        # prepare everything with accelerator
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
            model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
        )

        # save states
        if hasattr(args.checkpointing_steps, "isdigit"):
            checkpointing_steps = args.checkpointing_steps
            if args.checkpointing_steps.isdigit():
                checkpointing_steps = int(args.checkpointing_steps)
        else:
            checkpointing_steps = None

    # metric
    #metric = load_metric('sacrebleu')

    # train and eval
    if args.mode == "train":
        total_batch_size = args.per_device_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {len(train_dataset)}")
        logger.info(f"  Num Epochs = {args.num_train_epochs}")
        logger.info(f"  Instantaneous batch size per device = {args.per_device_batch_size}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {args.max_train_steps}")
        # Only show the progress bar once on each machine
        progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
        completed_steps = 0
        for epoch in range(args.num_train_epochs):
            model.train()
            for step, batch in enumerate(train_dataloader):
                # We need to skip steps until we reach the resumed step
                outputs = model(**batch)
                loss = outputs['loss']
                # We keep track of the loss at each epoch
                loss = loss / args.gradient_accumulation_steps
                accelerator.backward(loss)
                if step % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                    progress_bar.update(1)
                    completed_steps += 1
                    wandb.log({'loss':loss}, step=completed_steps)

                if isinstance(checkpointing_steps, int):
                    if completed_steps % checkpointing_steps == 0:
                        output_dir = f"step_{completed_steps}"
                        if args.output_dir is not None:
                            output_dir = os.path.join(args.output_dir, output_dir)
                        accelerator.save_state(output_dir)
                    
                if completed_steps >= args.max_train_steps:
                    break
            model.eval()
            gen_kwargs = {
                "max_length": args.max_target_length if args is not None else config.max_length,
                "num_beams": args.num_beams,
            }
            eval_preds = []
            eval_labels = []
            for step, batch in enumerate(eval_dataloader):
                with torch.no_grad():
                    generated_tokens = accelerator.unwrap_model(model).generate(
                        batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        **gen_kwargs,
                    )

                    generated_tokens = accelerator.pad_across_processes(
                        generated_tokens, dim=1, pad_index=tokenizer.pad_token_id
                    )
                    labels = batch["labels"]
                    labels = accelerator.pad_across_processes(batch["labels"], dim=1, pad_index=tokenizer.pad_token_id)

                    generated_tokens = accelerator.gather(generated_tokens).cpu().numpy()
                    labels = accelerator.gather(labels).cpu().numpy()

                    if isinstance(generated_tokens, tuple):
                        generated_tokens = generated_tokens[0]
                    decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
                    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
                    eval_preds.extend(decoded_preds)
                    eval_labels.extend(decoded_labels)
                    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
            
            #result = metric.compute()
            # 预处理需要计算metric的输入
            eval_preds_metric = []
            eval_refs_metric = []
            for i in range(0,len(eval_preds)):
                eval_preds_metric.append(eval_preds[i].split())
                eval_refs_metric.append(eval_labels[i].split())
            # 计算
            BLEU_1, BLEU_2 = compute_BLEU_batch(eval_preds_metric, eval_refs_metric)
            distinct_1, distinct_2 = compute_distinct_batch(eval_preds_metric)
            f1_score = compute_f1_batch(eval_preds_metric, eval_refs_metric)
            result = {'BLEU_1':BLEU_1, 'BLEU_2':BLEU_2, 'distinct_1':distinct_1, 'distinct_2':distinct_2, 'f1_score':f1_score}
            wandb.log(result)
            logger.info(result)
            
            eval_output = []
            for i in range(0,len(eval_preds)):
                eval_output.append(eval_preds[i].replace(" ", "") + '\t' + eval_labels[i].replace(" ", ""))

            test_output = []
            for src in raw_datasets['test']['input']:
                # encode方法把输入转为ids
                input_ids = tokenizer.encode(src, max_length=args.max_source_length, return_tensors='pt')
                logits = accelerator.unwrap_model(model).generate(input_ids.cuda(), num_beams=4,)
                tgt = tokenizer.decode(logits.squeeze(), skip_special_tokens=True)
                # 之前出现的问题：多卡下每一台都在打印信息，可以wait一下来解决
                test_output.append(tgt.replace(" ", ""))

            if args.checkpointing_steps == "epoch":
                output_dir = f"epoch_{epoch}"
                if args.output_dir is not None:
                    output_dir = os.path.join(args.output_dir, output_dir)
                accelerator.save_state(output_dir)
                with open(os.path.join(output_dir,'eval_output.txt'), "w",encoding='UTF-8') as writer:
                    writer.write("\n".join(eval_output))
                with open(os.path.join(output_dir,'test_output.txt'), "w",encoding='UTF-8') as writer:
                    writer.write("\n".join(test_output))
        # 所有epoch都结束
        wandb.finish()
        if args.output_dir is not None:
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(args.output_dir, save_function=accelerator.save)
            print("save model")
            if accelerator.is_main_process:
                tokenizer.save_pretrained(args.output_dir)
            with open(os.path.join(args.output_dir, "all_results.json"), "w") as f:
                json.dump(result, f)
    if args.mode == "test":
        """batch decode test set"""

if __name__ == "__main__":
    main()

