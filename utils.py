import os
import logging
from datetime import datetime
from config import Configuration

from datasets import load_metric

logger = logging.getLogger(__name__)

# Load the WER metric
wer_metric = load_metric("wer")


def compute_metrics(pred, config: Configuration):
    """
    Compute the Word Error Rate (WER) metrics.

    Args:
        pred: Predictions output from the model.
        config (Configuration): the training configuration
    Returns:
        A dictionary containing the WER metric.
    """
    logger.info("Computing metrics...")

    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # Replace -100 in the labels as they are not to be considered in metrics
    label_ids[label_ids == -100] = config.processor.tokenizer.pad_token_id

    pred_str = config.processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = config.processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    # Normalize predictions and references if the flag is set
    if config.do_normalize_eval:
        pred_str = [config.normalizer.normalize_text(pred) for pred in pred_str]
        label_str = [config.normalizer.normalize_text(label) for label in label_str]

    # Filter out empty references and their corresponding predictions
    valid_pred_str, valid_label_str, empty_references = filter_empty_references(pred_str, label_str)

    # Log the number of empty references found
    if empty_references:
        logger.warning(f"{len(empty_references)} empty references found, ignoring them for metric computation.")

    # Compute the Word Error Rate (WER)
    wer = wer_metric.compute(predictions=valid_pred_str, references=valid_label_str)
    return {"wer": wer}


def filter_empty_references(pred_str, label_str):
    """
    Filter out predictions and references that are empty.

    Args:
        pred_str (list of str): The list of predicted transcriptions.
        label_str (list of str): The list of ground truth transcriptions.

    Returns:
        valid_pred_str (list of str): The filtered list of predicted transcriptions.
        valid_label_str (list of str): The filtered list of ground truth transcriptions.
        empty_references (list of str): The list of empty references found.
    """
    valid_pred_str = []
    valid_label_str = []
    empty_references = []

    for pred, label in zip(pred_str, label_str):
        if len(label.strip()) == 0:
            empty_references.append(label)
        else:
            valid_pred_str.append(pred)
            valid_label_str.append(label)

    return valid_pred_str, valid_label_str, empty_references


# Set up logging in a standard format
def setup_logging(log_file=None, level=logging.INFO):
    """
    Set up logging configuration to output to both file and console.

    Args:
        log_file (str, optional): Path to a file where logs should be written. If None, logs are written only to standard output.
        level (int): Logging level, e.g., logging.INFO, logging.DEBUG
    """
    # Define a default log file if the log file is None
    log_file = log_file if log_file is not None else f"./logs/training_{datetime.now().strftime('%Y_%m_%d-%H_%M')}.log"

    # Define the format for the log messages
    log_format = "%(asctime)s - %(levelname)s - %(name)s -   %(message)s"
    date_format = "%m/%d/%Y %H:%M:%S"

    # Clear any previously configured handlers
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Configure root logger to the lowest level to delegate filter control to handlers
    root_logger.setLevel(level)

    # Create a console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(fmt=log_format, datefmt=date_format))
    console_handler.setLevel(level)  # Set the log level for the console
    root_logger.addHandler(console_handler)

    # Create a file handler if a log file is specified
    if log_file:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file, mode='a')
        file_handler.setFormatter(logging.Formatter(fmt=log_format, datefmt=date_format))
        file_handler.setLevel(level)  # Set the log level for the file
        root_logger.addHandler(file_handler)

    # Disable logging for some verbose modules used by huggingface transformers
    logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)


# Helper function to ensure a directory exists
def ensure_dir(directory):
    """
    Ensure that a directory exists.

    Args:
        directory (str): Path to the directory.
    """
    if not os.path.exists(directory):
        logging.info(f"Creating directory: {directory}")
        os.makedirs(directory)
