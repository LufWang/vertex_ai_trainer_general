# import pacakages
from pytorch_nlp_pipeline.model import PytorchNlpModel
from pytorch_nlp_pipeline.dataset import ClfDataset
from pytorch_nlp_pipeline.classification import Trainer
from pytorch_nlp_pipeline.utils import save_model
import numpy as np
import os
import json
import hypertune
import logging


WORKER = '[bold cyan]PIPELINE TRAIN[/bold cyan]'


def run_training_pipeline(df, df_val, text_col, label_col, **kwargs):

    logging.info(f'{WORKER}:Text Column {text_col}')
    logging.info(f'{WORKER}:Label Column {label_col}')

    ## load params_range
    MAX_LEN = kwargs['MAX_LEN']
    BATCH_SIZE = kwargs['BATCH_SIZE']
    RANDOM_SEED = kwargs['RANDOM_SEED']
    device = kwargs['DEVICE']
    pretrained_dir = kwargs['PRETRAINED_DIR']
    pretrained_model_name = kwargs['PRETRAINED_MODEL_NAME']

    pretrained_path = os.path.join(pretrained_dir, pretrained_model_name)
    
    ## Evaluate config
    save_path = kwargs['SAVE_PATH']
    save_metric = kwargs['SAVE_METRIC']
    multiclass_average = kwargs['MULTICLASS_AVERAGE']
    watch_metrics = kwargs['WATCH_METRICS']
    eval_freq = kwargs['EVAL_FREQ']
    focused_indexes = kwargs['focused_indexes']
    labels_to_indexes = kwargs['labels_to_indexes']
    save_mode = kwargs['save_mode']
    hyper_tune = kwargs['hyper_tune']

    ## Params
    epochs = kwargs['EPOCHS']
    lr = kwargs['LR']
    weight_decay = kwargs['WEIGHT_DECAY']
    warmup_steps = kwargs['WARMUP_STEPS']

    # Model config
    freeze_pretrained = kwargs['FREEZE_PRETRAINED']
    head_hidden_layers = kwargs['HEAD_HIDDEN_LAYERS']
    model_id = kwargs['model_id']

    eval_config = {
            "save_metric": save_metric,
            "multiclass_average": multiclass_average,
            "focused_indexes": focused_indexes,
            "eval_freq": eval_freq,
            "watch_list": watch_metrics
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

    clf = PytorchNlpModel(
                        pretrained_type='BERT',
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

    model_info['pretrained_model_name'] = pretrained_model_name
    model_info['head_hidden_layers'] = head_hidden_layers

    files = {
        'hyperparameters.json': params,
        'model_info.json': model_info,
        'labels_to_indexes.json': labels_to_indexes,
        'indexes_to_labels.json': train_data.indexes_to_labels
    }
    save_model(model, model_id,clf.tokenizer, label_col, save_path, files, save_mode)

    val_score = model_info['val_score']
    logging.info(f'{WORKER}: Model Saved -- Val Score {val_score}')
    
    if hyper_tune:
        logging.info(f'{WORKER}:Reporting score to hypertune -- metric_name:{save_metric} Score{val_score}')
        hpt = hypertune.HyperTune()
        hpt.report_hyperparameter_tuning_metric(
            hyperparameter_metric_tag=save_metric,
            metric_value=val_score,
            global_step=model_info['epoch'])