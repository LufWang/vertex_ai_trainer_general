# import pacakages
from pytorch_nlp_pipeline.ModelModule import BertModule, BioGPTModule
from pytorch_nlp_pipeline.DataModule import ClfDataset, split_data_w_sample
from pytorch_nlp_pipeline.text_clf import Trainer
from pytorch_nlp_pipeline.utils import get_config, save_model
import numpy as np
import os
import json

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
    
    save_path = kwargs['SAVE_PATH']
    save_metric = kwargs['SAVE_METRIC']
    multiclass_average = kwargs['MULTICLASS_AVERAGE']
    watch_metrics = kwargs['WATCH_METRICS']
    eval_freq = kwargs['EVAL_FREQ']
    epochs = kwargs['EPOCHS']
    lr = kwargs['LR_RANGE']
    weight_decay = kwargs['WEIGHT_DECAY_RANGE']
    warmup_steps = kwargs['WARMUP_STEPS']
    save_threshold = kwargs['SAVE_THRESHOLD']

    focused_indexes = kwargs['focused_indexes']
    labels_to_indexes = kwargs['labels_to_indexes']

    ModelModule = kwargs['MODELMODULE']
    freeze_pretrained = kwargs['FREEZE_PRETRAINED']



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


    clf = ModelModule(pretrained_path=pretrained_path, freeze_pretrained=freeze_pretrained)


    BertTrainer = Trainer(device)


    ##########
    # Training
    ##########
    params = {
        "lr": lr,
        "weight_decay": weight_decay,
        "EPOCHS": epochs,
        "warmip_steps": warmup_steps
    }


    model, model_info = BertTrainer.train(clf, train_data, val_data, params, eval_config)
    
    logging.info(f'{WORKER}: End of Training Result:')
    logging.info(f'{WORKER}: {model_info}')
    val_score = model_info['val_score']                       

    if val_score > save_threshold:
        
        model_info['pretrained_model_name'] = pretrained_model_name

        files = {
        'hyperparameters.json': params,
        'model_info.json': model_info,
        'labels_to_indexes.json': labels_to_indexes,
        'indexes_to_labels.json': train_data.indexes_to_labels
                    
                }
        save_model(model, clf.tokenizer, label_col, save_path, files)

        #TODO log saved results?
        logging.info(f'{WORKER}: Model Saved -- Val Score {val_score}')