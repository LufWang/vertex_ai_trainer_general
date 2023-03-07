from functools import wraps
import logging
import rich
from rich.logging import RichHandler
from datetime import datetime
from config import LOG_DIR
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

def prep_log(is_debug):
    WORKER = '[bold]LOGGER[/bold]'
    now = datetime.now()
    time_stamp = now.strftime("%Y-%m-%d-%H-%M-%S")
    log_path = os.path.join(LOG_DIR, "RUN_"+str(time_stamp)+'.log')
    #handlers = [logging.StreamHandler(sys.stdout)]
    rich_handler = RichHandler(markup=True, highlighter=rich.highlighter.NullHighlighter())
    if is_debug:
        rich_handler.setLevel(logging.DEBUG)
    else:
        rich_handler.setLevel(logging.INFO)
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.DEBUG)
    handlers = [rich_handler, file_handler]
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(message)s",
        # format="%(message)s",
        handlers=handlers)
    logging.info(f"{WORKER}: log path ([green]{log_path}[/green])")