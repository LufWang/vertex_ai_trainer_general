from functools import wraps
import logging
import google.cloud.logging

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
    # WORKER = '[bold]LOGGER[/bold]'
    now = datetime.now()
    time_stamp = now.strftime("%Y-%m-%d-%H-%M-%S")
    log_path = os.path.join(LOG_DIR, "TRAIN_"+str(time_stamp)+'_'+model_id+'.log')
    # handlers = [logging.StreamHandler(sys.stdout)]
    # rich_handler = RichHandler(markup=True, highlighter=rich.highlighter.NullHighlighter())
    

    client = google.cloud.logging.Client()

    # Retrieves a Cloud Logging handler based on the environment
    # you're running in and integrates the handler with the
    # Python logging module. By default this captures all logs
    # at INFO level and higher
    client.setup_logging()

    logging.basicConfig(filename=log_path, 
                        encoding='utf-8', 
                        format="%(message)s",
                        level=logging.DEBUG)



    logging.info("Vertex AI Trainer Logger Initiated")

    # logging.info(f"{WORKER}: log path ([green]{log_path}[/green])")