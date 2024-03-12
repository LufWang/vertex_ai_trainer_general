import argparse



import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score
import json
import shortuuid
from datasets import load_dataset
from transformers import DataCollatorWithPadding
from transformers import AutoTokenizer
import evaluate
import numpy as np
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from torch import nn
import torch



import os

from utils import parse_env_bool

import shortuuid

#######
# Set up Logging
########
import google.cloud.logging_v2 as logging_v2
import logging
client = logging_v2.client.Client()
# set the format for the log
google_log_format= logging.Formatter(
fmt='%(name)s | %(module)s | %(funcName)s | %(message)s',
datefmt='%Y-%m-$dT%H:%M:%S')


handler = client.get_default_handler()
handler.setFormatter(google_log_format)

cloud_logger = logging.getLogger("vertex-ai-job-logger")
cloud_logger.setLevel("INFO")
cloud_logger.addHandler(handler)


log = logging.getLogger("vertex-ai-job-logger")

WORKER = 'PIPELINE MAIN'

  

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


parser.add_argument('--hyper_tune', dest='hyper_tune', 
                        help='whether run vertex ai hypertune', default=False, type=parse_env_bool)


parser.add_argument('--train_file_path', dest='train_file_path', 
                        help='train file path ', type=str, required=True)

parser.add_argument('--val_file_path', dest='val_file_path', 
                        help='val file path ', type=str, required=True)

parser.add_argument('--test_file_path', dest='test_file_path', 
                        help='test file path ', type=str, required=True)

parser.add_argument('--save_path', dest='save_path', 
                        help='save path for model artifacts', type=str, required=False)

parser.add_argument('--eval_freq', dest='eval_freq', 
                        help='eval frequency per epoch', type=int, default=1)

parser.add_argument('--max_len', dest='max_len', 
                        help='max length for each text input', type=int, default=156)

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

parser.add_argument('--gcp_project_id', dest='gcp_project_id', 
                        help='GCP project id', type=str, required=False, default=None)


args = parser.parse_args()


### Setting Variables

args = vars(args)

model_id = shortuuid.ShortUUID().random(length=12)
args['model_id'] = model_id


## List out config 
for key in args:
    log.info(f'{WORKER}: {key}: {args[key]}')


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
log.info(f'DEVICE: {device}')

#################################################################################################################
## Getting data 
dataset = load_dataset("csv", data_files=args['train_file_path'])
dataset_val = load_dataset("csv", data_files=args['val_file_path'])
dataset_test = load_dataset("csv", data_files=args['test_file_path'])

tokenizer = AutoTokenizer.from_pretrained(args['pretrained_path'])
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

def text_preprocess_function(examples):
    return tokenizer(examples[args['text_col']], truncation=True, max_length=args['max_len'])

tokenized_data = dataset.map(text_preprocess_function, batched=True)
tokenized_data_val = dataset_val.map(text_preprocess_function, batched=True)

labels = set(dataset['train'][args['label_col']] + 
             dataset_val['train'][args['label_col']] +
             dataset_test['train'][args['label_col']]
             )

labels_to_indexes = {v: k for k, v in enumerate(labels)}
indexes_to_labels = {v:k for k,v in labels_to_indexes.items()}

## List to label to indexes 
log.info('Labels to Indexes:')
for key in labels_to_indexes:
    log.info(f'{WORKER}: {key}: {labels_to_indexes[key]}')

focused_labels = args.get('focused_labels', None)
log.info(f'focused_labels: {focused_labels}')
if focused_labels:
    focused_indexes = [labels_to_indexes[label] for label in focused_labels]
else:
    focused_indexes = None
log.info(f'focused_ indexes: {focused_indexes}')


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    f1 = f1_score(labels, predictions, labels=focused_indexes,average='weighted')
    precision = precision_score(labels, predictions, labels=focused_indexes,average='weighted')
    recall = recall_score(labels, predictions, labels=focused_indexes,average='weighted')

    return {'f1': f1, 'precision': precision, 'recall': recall}

def preprocess_function(datapoint):
    datapoint['label'] = labels_to_indexes[datapoint[args['label_col']]]
    return datapoint

## prepare dataset for training
tokenized_data = tokenized_data.map(preprocess_function)
tokenized_data_val = tokenized_data_val.map(preprocess_function)

model = AutoModelForSequenceClassification.from_pretrained(
    args['pretrained_path'], 
    num_labels=len(labels_to_indexes), 
    id2label=indexes_to_labels, 
    label2id=labels_to_indexes
)

class_weight = [] 
label_counts= pd.Series(dataset['train'][args['label_col']] + 
                        dataset_val['train'][args['label_col']] + 
                        dataset_test['train'][args['label_col']]).value_counts()


for label in labels_to_indexes:
    class_weight.append(float(max(label_counts.values) / label_counts[label]))

log.info(f'{WORKER}: class weights: {class_weight}')
    
loss_fn = nn.CrossEntropyLoss(
                            weight = torch.tensor(class_weight)
                            ).to(device)

class DFDTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs[0]
        loss = loss_fn(logits, labels)
        
        return (loss, outputs) if return_outputs else loss




#######
# Training
#######
model_out_path = args['save_path']

os.environ["TOKENIZERS_PARALLELISM"] = "false"
training_args = TrainingArguments(
    output_dir='checkpoints',
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=4,
    weight_decay=0.01,
    evaluation_strategy="steps",
    eval_steps=2000,
    save_strategy="steps",
    save_steps=2000,
    load_best_model_at_end=True,
    dataloader_num_workers=2,
    dataloader_prefetch_factor=2,
    metric_for_best_model='f1',
    greater_is_better=True,
    log_level='info'
    
)

model = model.to(device)

trainer = DFDTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_data['train'],
    eval_dataset=tokenized_data_val['train'],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

trainer.train()

### save best model 
trainer.save_model(model_out_path)

# Run predictions 
dataset_all = load_dataset("csv", data_files='all.csv')
tokenized_dataset_all = dataset_all.map(text_preprocess_function, batched=True)
tokenized_dataset_all = tokenized_dataset_all.map(preprocess_function)
predictions = trainer.predict(tokenized_dataset_all['train'])

m = nn.Softmax(dim=1)
preds_proba, preds = torch.max(m(torch.tensor(predictions.predictions)), dim=1)


df_pred = pd.DataFrame({
        "text": tokenized_dataset_all['train']['cleaned_text'],
        "label": tokenized_dataset_all['train']['new_label'],
        "pred": [indexes_to_labels[pred] for pred in preds.tolist()]
})

df_pred.to_csv(os.path.join(model_out_path, f'{model_id}-predictions.csv'))

## generate predictions.json
predictions_json = {}
texts = tokenized_dataset_all['train']['cleaned_text']
pred_proba_all = predictions.predictions
m = nn.Softmax(dim=0)
for i in range(len(texts)):
    pred_proba = pred_proba_all[i]
    pred_proba = m(torch.tensor(pred_proba)).tolist()

    predictions_json[texts[i]] = pred_proba

with open(os.path.join(model_out_path, f'{model_id}-eval_predictions.json'), 'w') as f:
    json.dump(predictions_json, f)
    
with open(os.path.join(model_out_path, f'{model_id}-labels_to_indexes.json'), 'w') as f:
    json.dump(labels_to_indexes, f)

with open(os.path.join(model_out_path, f'{model_id}-indexes_to_labels.json'), 'w') as f:
    json.dump(indexes_to_labels, f)