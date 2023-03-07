from sklearn.metrics import precision_score, recall_score, confusion_matrix, f1_score, accuracy_score, precision_recall_fscore_support
from pytorch_nlp_pipeline.ModelModule import BertModule, BioGPTModule

LOG_DIR = 'logs'

training_config = {
    "RANDOM_SEED": 42,
    "BATCH_SIZE": 16,
    "MAX_LEN": 162,
    # "PRETRAINED_DIR": "/home/ext_wang_lufei_mayo_edu/LFW/Data_Science_Model_Training/gen4/storage/gen4-models/pretrained-models",
    "PRETRAINED_DIR": "/gcs/ml-dd9a-phi-shared-aif-us-p/gen4-models/pretrained-models",
    "PRETRAINED_MODEL_NAME": 'pretrained_bioclinical_bert_on_all_dtree',
    "FREEZE_PRETRAINED": False,
    "MODELMODULE": BertModule,
    "EVAL_FREQ": 3,
    "EPOCHS": [12],
    "LR_RANGE": [1e-5, 5e-5],
    "WEIGHT_DECAY_RANGE": [1e-2, 1e-5],
    "WARMUP_STEPS": [500]
}

dataset_config = {
    "dataset_dir_path": "gs://ml-dd9a-phi-shared-aif-us-p/gen4-models/data",
    "version": "0.0.1"
}

pipeline_config = {
    "SAVE_PATH": "/gcs/ml-dd9a-phi-shared-aif-us-p/gen4-models/trained-models-v2/dtree-all-models",
    "SAVE_METRIC": f1_score,
    "MULTICLASS_AVERAGE": "weighted",
    "WATCH_METRICS": {
        "f1_score": f1_score,
        "precision": precision_score,
        'recall': recall_score
    },
    "NUM_ITERS": 10,
    "SAVE_THRESHOLD": 0.4
    
}

