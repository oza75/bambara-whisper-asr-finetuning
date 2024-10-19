import logging

from transformers.models.whisper.english_normalizer import BasicTextNormalizer

from config import parse_args, Configuration
from data_preparation import load_and_prepare_dataset
from model_setup import setup_model_and_processor, DataCollatorSpeechSeq2SeqWithPadding
from utils import setup_logging


def main():
    # Parse command line arguments
    args = parse_args()

    # Set up logging
    setup_logging()

    # Load configuration from parsed arguments
    config = Configuration(args)
    print(args)
    # Set up the model and processor
    logging.info("Setting up model and processor...")
    model, processor, feature_extractor, tokenizer = setup_model_and_processor(config)

    config.processor = processor
    config.feature_extractor = feature_extractor
    config.tokenizer = tokenizer
    config.normalizer = BasicTextNormalizer()

    # Prepare dataset
    logging.info("Loading and preparing datasets...")
    datasets = load_and_prepare_dataset(config)

    datasets['train'].push_to_hub("oza75/bambara-asr-preprocessed", split="train")
    datasets['test'].push_to_hub("oza75/bambara-asr-preprocessed", split="test")


if __name__ == "__main__":
    main()
