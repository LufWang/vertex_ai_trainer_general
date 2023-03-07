import sys
import argparse


# import pacakages
import torch
import numpy as np
from importlib import reload
import os
from datetime import datetime
import json

from config import training_config, dataset_config, pipeline_config
from data_fetch import get_train_test
from training import train
from training.data_preprocess import preprocess_binary_clf, preprocess_multi_clf

import logging
from utils import prep_log




#################################################################################################################
## Arguments


parser = argparse.ArgumentParser(description='General Training Pipeline for Text Classification')

parser.add_argument('--mode', dest='mode', 
                        help='Training Mode: binary / multi', required=True, choices=['binary', 'multi'])

parser.add_argument('--text_col', dest='text_col', 
                        help='Text Column Name', required=True)

parser.add_argument('--label_col', dest='label_col', 
                        help='Label Column Name', required=True)

parser.add_argument('--label', dest='label', 
                        help='Required if mode==binary, will train binary for this label')

parser.add_argument('--focused_label', dest='focused_label', nargs='+',
                        help='focused laels for for multiclf --optional -- will be ignored if mode==binary')

parser.add_argument('--EPOCH', dest='EPOCH', 
                        help='how many epochs to run', default=4)

parser.add_argument('--learning_rate', dest='learning_rate', 
                        help='how many epochs to run', default=0.0001)

parser.add_argument('--weight_decay', dest='weight_decay', 
                        help='how many epochs to run', default=0.001)

parser.add_argument('--warmup_steps', dest='warmup_steps', 
                        help='how many epochs to run', default=0)



args = parser.parse_args()

prep_log(False)
WORKER = '[bold cyan]PIPELINE MAIN[/bold cyan]'

### Setting Variables
RANDOM_SEED = training_config['RANDOM_SEED']
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
training_config['DEVICE'] = device


# file names
dataset_version = dataset_config['version']
dataset_dir = dataset_config['dataset_dir_path']
train_file_path = os.path.join(dataset_dir, f'DFD-data-{dataset_version}/train.csv')
val_file_path = os.path.join(dataset_dir, f'DFD-data-{dataset_version}/val.csv')



## List out config 
logging.info(f'{WORKER}: Training Config:')
for key in training_config:
    logging.info(f'{WORKER}: {key}: {training_config[key]}')

logging.info(f'{WORKER}: Dataset Config:')
for key in dataset_config:
    logging.info(f'{WORKER}: {key}: {dataset_config[key]}')

logging.info(f'{WORKER}: Pipeline Config:')
for key in pipeline_config:
    logging.info(f'{WORKER}: {key}: {pipeline_config[key]}')


#################################################################################################################
## Getting data 


# can change to get using GCS Python client?
df_train, df_val = get_train_test(train_file_path, val_file_path)

logging.info(f'{WORKER}: Train Data Shape: {df_train.shape}')
logging.info(f'{WORKER}: Val Data Shape: {df_val.shape}')




#############################################################################################################################
# Training pipeline

training_config['EPOCHS'] = args.EPOCH
training_config['lr'] = args.learning_rate
training_config['weight_decay'] = args.weight_decay
training_config['warmup_steps'] = args.warmup_steps



input_dict = {
    **training_config, 
    **pipeline_config
    }


if args.mode == 'binary':
    if args.label:
        logging.info(f'{WORKER}: Training Binary Clf on {args.label}...')
        df_train, df_val, label_col, labels_to_indexes = preprocess_binary_clf(df_train, df_val, args.label_col, args.label)
        input_dict['labels_to_indexes'] = labels_to_indexes
        input_dict['focused_indexes'] = None
        train.run_training_pipeline(df_train, df_val, args.text_col, label_col, **input_dict)
    
    else:
        logging.info(f'{WORKER}: Need to pass in --label when --mode==binary')

elif args.mode == 'multi':
    logging.info(f'{WORKER}: Training Multi Clf on {args.label}...')
    df_train, df_val, label_col, labels_to_indexes, focused_indexes = preprocess_multi_clf(df_train, df_val, args.label_col, args.focused_label)

    input_dict['labels_to_indexes'] = labels_to_indexes
    input_dict['focused_indexes'] = focused_indexes
    train.run_training_pipeline(df_train, df_val, args.text_col, label_col, **input_dict)


