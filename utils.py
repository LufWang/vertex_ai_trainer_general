from functools import wraps
import logging
from datetime import datetime
import argparse
import os

from sklearn.metrics import precision_score, recall_score, f1_score


def suspend_logging(func):
    @wraps(func)
    def inner(*args, **kwargs):
        logging.disable(logging.FATAL)
        try:
            return func(*args, **kwargs)
        finally:
            logging.disable(logging.NOTSET)
    return inner



def parse_env_bool(var_val):
    """
    Parse true or false values from the environment.
    True values are fuzzy-matched so they can be TRUE, True, true, t, and a variety of other possible values.
    This helps to adapt to env vars that may be shared with other systems.
    """
    if var_val is True or str(var_val).lower() in ('true', 't', 'yes', 'y', '1'):
        return True
    elif var_val.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')




