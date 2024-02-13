import pandas as pd
import logging
from functools import wraps


def suspend_logging(func):
    @wraps(func)
    def inner(*args, **kwargs):
        logging.disable(logging.FATAL)
        try:
            return func(*args, **kwargs)
        finally:
            logging.disable(logging.NOTSET)
    return inner

@suspend_logging
def get_data(train_file_path, val_file_path):
    df_train = pd.read_csv(train_file_path, low_memory = False)
    df_val = pd.read_csv(val_file_path, low_memory = False)



    return df_train, df_val