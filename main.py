import argparse

# import pacakages
import numpy as np

from data_fetch import get_data
from training import train
from training.data_preprocess import preprocess_binary_clf, preprocess_multi_clf

import logging

from utils import parse_env_bool

import shortuuid

#######
# Set up Logging
########
WORKER = 'PIPELINE MAIN'
logging.basicConfig(encoding='utf-8', level=logging.DEBUG)

  

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

parser.add_argument('--focused_label', dest='focused_label', action='append',
                        help='focused laels for for multiclf --optional -- will be ignored if mode==binary')

parser.add_argument('--pretrained_path', dest='pretrained_path',
                        help='path to folder that stores the pretrained model', required=True)

parser.add_argument('--pretrained_type', dest='pretrained_type',
                        help='type of pretrained model - BERT/Biogpt/ALBERT ', default='BERT')

parser.add_argument('--num_epochs', dest='num_epochs', 
                        help='how many epochs to run', default=1, type=int)

parser.add_argument('--learning_rate', dest='learning_rate', 
                        help='how many epochs to run', default=0.0001, type=float)

parser.add_argument('--weight_decay', dest='weight_decay', 
                        help='how many epochs to run', default=0.001, type=float)

parser.add_argument('--warmup_steps', dest='warmup_steps', 
                        help='how many epochs to run', default=0, type=int)

parser.add_argument('--batch_size', dest='batch_size', 
                        help='batch size', default=16, type=int)

parser.add_argument('--save_mode', dest='save_mode', 
                        help='save whole model, just body or just head', choices=['body-only', 'head-only'])

parser.add_argument('--hyper_tune', dest='hyper_tune', 
                        help='whether run vertex ai hypertune', default=False, type=parse_env_bool)

parser.add_argument('--freeze_pretrained', dest='freeze_pretrained', 
                        help='whether freeze pretrained part', default=False, type=parse_env_bool)

parser.add_argument('--env', dest='env', 
                        help='path to .env file that stores env variables', type=str)

parser.add_argument('--dataset_path', dest='train_file_path', 
                        help='train file path ', type=str, required=True)

parser.add_argument('--train_file_path', dest='train_file_path', 
                        help='train file path ', type=str, required=True)

parser.add_argument('--val_file_path', dest='val_file_path', 
                        help='val file path ', type=str, required=True)

parser.add_argument('--save_path', dest='save_path', 
                        help='save path for model artifacts', type=str, required=False)

parser.add_argument('--eval_freq', dest='eval_freq', 
                        help='eval frequency per epoch', type=int, default=1)

parser.add_argument('--max_len', dest='max_len', 
                        help='max length for each text input', type=int, default=162)

parser.add_argument('--random_seed', dest='random_seed', 
                        help='save path for model artifacts', type=int, default=42)

parser.add_argument('--save_metric', dest='save_metric', 
                        help='save metric for checkpointing models', type=str, default='f1')

parser.add_argument('--multiclass_average', dest='multiclass_average', 
                        help='average method to use if calculating metric for multiclass', type=str, default='weighted')

parser.add_argument('--model_cat_uid', dest='model_cat_uid', 
                        help='trained model category unique identifier', type=str, required=False)

parser.add_argument('--job_id', dest='job_id', 
                        help='Option to pass in job id to be put in to BQ (Custom Assigned - not the GCP job id)', type=str, required=False)

parser.add_argument('--bq_table', dest='bq_table', 
                        help='bq table name to save trained model info', type=str, required=False)

parser.add_argument('--dataset_version', dest='dataset_version', 
                        help='dataset version identifier', type=str, required=True)


args = parser.parse_args()


### Setting Variables

input_dict = vars(args)

model_id = shortuuid.ShortUUID().random(length=12)
input_dict['model_id'] = model_id


## List out config 
for key in input_dict:
    logging.info(f'{WORKER}: {key}: {input_dict[key]}')



#################################################################################################################
## Getting data 
df_train, df_val = get_data(args.train_file_path, args.val_file_path)


logging.info(f'{WORKER}: Train Data Shape: {df_train.shape}')
logging.info(f'{WORKER}: Val Data Shape: {df_val.shape}')


#############################################################################################################################
# Training pipeline

if args.mode == 'binary':
    if args.label:
        logging.info(f'{WORKER}: Training Binary Clf on {args.label}...')
        df_train, df_val, label_col, labels_to_indexes = preprocess_binary_clf(df_train, df_val, args.label_col, args.label)
        input_dict['labels_to_indexes'] = labels_to_indexes
        input_dict['focused_indexes'] = None
        input_dict['label_col'] = label_col
        train.run_training_pipeline(df_train, df_val, **input_dict)
    else:
        logging.info(f'{WORKER}: Need to pass in --label when --mode==binary')

elif args.mode == 'multi':
    logging.info(f'{WORKER}: Training Multi Clf on {args.label_col}...')
    df_train, df_val, label_col, labels_to_indexes, focused_indexes = preprocess_multi_clf(df_train, df_val, args.label_col, args.focused_label)

    input_dict['labels_to_indexes'] = labels_to_indexes
    input_dict['focused_indexes'] = focused_indexes
    input_dict['label_col'] = label_col
    train.run_training_pipeline(df_train, df_val, **input_dict)