# import pacakages
from pytorch_nlp_pipeline.model import TransformerNN
from pytorch_nlp_pipeline.dataset import ClfDataset
from pytorch_nlp_pipeline.classification import Trainer, Evaluator
from pytorch_nlp_pipeline.utils import save_model
import torch
import numpy as np
import pandas as pd
import os
import json
import hypertune
import logging
from datetime import datetime

from google.cloud import bigquery


WORKER = 'PIPELINE TRAIN'


def run_training_pipeline(df, df_val, **kwargs):

    text_col = kwargs['text_col']
    label_col = kwargs['label_col']

    logging.info(f'{WORKER}:Text Column {text_col}')
    logging.info(f'{WORKER}:Label Column {label_col}')

    ## load params_range
    MAX_LEN = kwargs['max_len']
    BATCH_SIZE = kwargs['batch_size']
    RANDOM_SEED = kwargs['random_seed']
    
    # set random seeds
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    pretrained_path = kwargs['pretrained_path']
    pretrained_type = kwargs['pretrained_type']
    dataset_version = kwargs['dataset_version']
    
    ## Evaluate config
    save_path = kwargs['save_path']
    save_metric = kwargs['save_metric']
    multiclass_average = kwargs['multiclass_average']
    eval_freq = kwargs['eval_freq']
    focused_indexes = kwargs['focused_indexes']
    labels_to_indexes = kwargs['labels_to_indexes']
    save_mode = kwargs['save_mode']
    hyper_tune = kwargs['hyper_tune']
    model_cat_uid = kwargs['model_cat_uid']


    if not model_cat_uid:
        model_cat_uid = label_col

    ## Params
    epochs = kwargs['num_epochs']
    lr = kwargs['learning_rate']
    weight_decay = kwargs['weight_decay']
    warmup_steps = kwargs['warmup_steps']

    # Model config
    freeze_pretrained = kwargs['freeze_pretrained']
    head_hidden_layers = []
    model_id = kwargs['model_id']

    logging.info(f'{WORKER}:Head Hidden Layers {head_hidden_layers}')

    eval_config = {
            "save_metric": save_metric,
            "multiclass_average": multiclass_average,
            "focused_indexes": focused_indexes,
            "eval_freq": eval_freq,
            "watch_list": ['f1', 'precision', 'recall']
            }

    train_data = ClfDataset(
                                df,
                                text_col,
                                label_col,
                                labels_to_indexes,
                                BATCH_SIZE,
                                MAX_LEN,
                                RANDOM_SEED
                            )

    val_data = ClfDataset(
                                df_val,
                                text_col,
                                label_col,
                                labels_to_indexes,
                                BATCH_SIZE,
                                MAX_LEN,
                                RANDOM_SEED
                            )

    clf = TransformerNN(
                        pretrained_type=pretrained_type,
                        pretrained_path=pretrained_path,
                        n_classes=len(labels_to_indexes),
                        freeze_pretrained=freeze_pretrained,
                        head_hidden_layers=head_hidden_layers
                        )

    BertTrainer = Trainer(device)

    ##########
    # Training
    ##########
    params = {
        "lr": lr,
        "weight_decay": weight_decay,
        "EPOCHS": epochs,
        "warmup_steps": warmup_steps
    }

    model, model_info = BertTrainer.train(clf, train_data, val_data, params, eval_config)
    
    logging.info(f'{WORKER}: End of Training Result:')
    logging.info(f'{WORKER}: {model_info}')

    if model_info:

        model_info['pretrained_path'] = pretrained_path
        model_info['head_hidden_layers'] = []
        model_info['train_file_path'] = kwargs['train_file_path']
        model_info['val_file_path'] = kwargs['val_file_path']

        files = {
            'hyperparameters.json': params,
            'model_info.json': model_info,
            'labels_to_indexes.json': labels_to_indexes,
            'indexes_to_labels.json': train_data.indexes_to_labels
        }
        save_model(model, model_id, clf.tokenizer, model_cat_uid, save_path, files, save_mode)

        val_score = model_info['val_score']
        model_folder = os.path.join(save_path, model_cat_uid + '-' + model_id)


        bq_table = kwargs['bq_table']
        job_id = kwargs['job_id']
        if bq_table and job_id:
            # log saved model to bq table
            logging.info(f'{WORKER}: Inserting Model Info into BQ table...')
            bqclient = bigquery.Client(project=kwargs['gcp_project_id'])

            query = f"""
                    INSERT
                        INTO `{bq_table}`

                        VALUES ('{model_id}', 
                                '{job_id}', 
                                '{model_cat_uid}', 
                                '{datetime.now()}', 
                                '{save_metric}', 
                                {val_score}, 
                                '{model_folder}',
                                '{os.path.basename(pretrained_path)}',
                                '{dataset_version}',
                                NULL
                                
                                )

                    """

            results = bqclient.query(query)


        logging.info(f'{WORKER}: Model Saved -- Val Score {val_score}')
        logging.info(f'{WORKER}: Model Saved -- Val Score {val_score}')

        

        # reporting score to gcp hypertune
        if hyper_tune:
            logging.info(f'{WORKER}:Reporting score to hypertune -- metric_name:{save_metric} Score{val_score}')
            hpt = hypertune.HyperTune()
            hpt.report_hyperparameter_tuning_metric(
                hyperparameter_metric_tag=save_metric,
                metric_value=val_score,
                global_step=model_info['epoch']
                )

            logging.info(f'{WORKER}: Model Checkpoint Path {model_folder}')