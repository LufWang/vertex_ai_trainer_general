from functools import wraps
import logging
import google.cloud.logging
import rich
from rich.logging import RichHandler

from datetime import datetime
import os

def suspend_logging(func):
    @wraps(func)
    def inner(*args, **kwargs):
        logging.disable(logging.FATAL)
        try:
            return func(*args, **kwargs)
        finally:
            logging.disable(logging.NOTSET)
    return inner

def prep_log(LOG_DIR, model_id):

    rich_handler = RichHandler(markup=False, highlighter=rich.highlighter.NullHighlighter())

    rich_handler.setLevel(logging.INFO)

    handlers = [rich_handler]
 
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=handlers)



    logging.info("Vertex AI Trainer Logger Initiated")



