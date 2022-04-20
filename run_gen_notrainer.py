import argparse
import json
import logging
import os
import random
import sys

import numpy as np
import torch
import transformers
from transformers import (BertTokenizer, HfArgumentParser,DataCollatorForSeq2Seq,Seq2SeqTrainer,
                          Seq2SeqTrainingArguments, TrainerCallback, AutoModelForSeq2SeqLM)
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