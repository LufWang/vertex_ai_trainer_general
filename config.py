import os
from dotenv import load_dotenv


print(os.getenv('LOG_DIR'))


def parse_env_bool(var_val):
    """
    Parse true or false values from the environment.
    True values are fuzzy-matched so they can be TRUE, True, true, t, and a variety of other possible values.
    This helps to adapt to env vars that may be shared with other systems.
    """
    if var_val is True or str(var_val).lower() in ('true', 't', 'yes', 'y', '1'):
        return True
    else:
        return False
    

def get_config(env_path):
    load_dotenv(env_path)

    LOG_DIR = os.getenv('LOG_DIR')

    training_config = {
        "RANDOM_SEED": os.getenv('RANDOM_SEED', 42),
        "BATCH_SIZE": os.getenv('BATCH_SIZE', 16),
        "MAX_LEN": os.getenv('MAX_LEN', 162),
    #  "PRETRAINED_DIR": "/home/ext_wang_lufei_mayo_edu/LFW/Data_Science_Model_Training/gen4/storage/gen4-models/pretrained-models",
        "PRETRAINED_DIR": os.getenv('PRETRAINED_DIR'),
        "PRETRAINED_MODEL_NAME": os.getenv("PRETRAINED_MODEL_NAME", 'pretrained_bioclinical_bert_on_all_dtree'),
        "EVAL_FREQ": os.getenv('EVAL_FREQ', 1),
        "HEAD_HIDDEN_LAYERS": os.getenv('HEAD_HIDDEN_LAYERS', [384, 'relu'])
    }

    dataset_config = {
        "DATASET_DIR": os.getenv("DATASET_DIR"),
        "TRAIN_FILE_PATH": os.getenv("TRAIN_FILE_PATH", "train.csv"),
        "VAL_FILE_PATH": os.getenv("VAL_FILE_PATH", "val.csv")
    }

    pipeline_config = {
        "SAVE_PATH": os.getenv("SAVE_PATH"),
        # SAVE_METRIC and MATCH METRICS should be strings that can be used in sklearn.metrics.get_scorer()
        "SAVE_METRIC": os.getenv('SAVE_METRIC', 'f1'),
        "WATCH_METRICS": ["f1", "precision", "recall"],
        "MULTICLASS_AVERAGE": os.getenv("MULTICLASS_AVERAGE", "weighted")
    }

    return LOG_DIR, training_config, dataset_config, pipeline_config