import pandas as pd
import numpy as np
import logging
import json


WORKER = 'PIPELINE DATA PREPROCESS'


def log_label_distribution(df, label_col, top=10):
    logging.info(f"{WORKER}: Label Distribution: ")
    vc = df[label_col].value_counts()
    vc_ind = vc.index
    vc_val = vc.values

    r = min(top, vc.shape[0])
    for i in range(r):
        logging.info(f'{WORKER}: {vc_ind[i]} | {vc_val[i]}')

    if r < vc.shape[0]:
        logging.info(f'{WORKER}: Lowest Label Count')
        logging.info(f'{WORKER}: {vc_ind[-1]} | {vc_val[-1]}')

def preprocess_binary_clf(df_train, df_val, label_col, label_name):
    df_train[label_name] = np.where(df_train[label_col] == label_name, 1, 0)
    df_val[label_name] = np.where(df_val[label_col] == label_name, 1, 0)

    label_col = label_name

    labels_to_indexes = {
            label_name: 1
        }

    log_label_distribution(df_train, label_col)

    return df_train, df_val, label_col, labels_to_indexes


def preprocess_multi_clf(df_train, df_val, label_col,focused_labels):

    log_label_distribution(df_train, label_col)


    labels = df_train[label_col].unique()


    labels_to_indexes = {v: k for k, v in enumerate(labels)}

    focused_indexes = []
    if focused_labels:
        focused_indexes = [labels_to_indexes[x] for x in focused_labels]


    df_train[label_col] = df_train[label_col].apply(lambda x: labels_to_indexes[x])
    df_val[label_col] = df_val[label_col].apply(lambda x: labels_to_indexes[x])


    return df_train, df_val, label_col,labels_to_indexes, focused_indexes






