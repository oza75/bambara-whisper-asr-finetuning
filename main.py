import logging

from transformers.models.whisper.english_normalizer import BasicTextNormalizer

from config import parse_args, Configuration
from data_preparation import load_and_prepare_dataset
from model_setup import setup_model_and_processor, DataCollatorSpeechSeq2SeqWithPadding
from training import create_trainer, run_training
from utils import setup_logging, compute_metrics


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

    # Loading the data collator
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id
    )

    # Create the trainer
    logging.info("Creating the trainer...")
    trainer = create_trainer(
        model=model,
        config=config,
        data_collator=data_collator,
        # compute_metrics=lambda pred: compute_metrics(pred, config),
        compute_metrics=None,
        train_dataset=datasets["train"],
        eval_dataset=datasets["test"]
    )

    # Run training
    logging.info("Starting training...")
    training_result = run_training(trainer)

    # Optionally, push to hub
    if config.push_to_hub:
        logging.info("Pushing the model to the Hugging Face Hub...")
        trainer.push_to_hub()

    # Final logging of training results
    logging.info(f"Training completed with results: {training_result}")


if __name__ == "__main__":
    main()
