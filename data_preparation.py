from datasets import load_dataset, Audio
from config import Configuration
import logging

logger = logging.getLogger(__name__)


def load_and_prepare_dataset(config: Configuration):
    """
    Load and prepare the Whisper dataset for training.

    Args:
        config (Configuration): The Configuration object containing all the settings.

    Returns:
        DatasetDict: A dictionary containing the processed 'train' and 'test' datasets.
    """
    logger.info(f"Loading the {config.dataset} dataset from the Hugging Face Hub...")
    common_voice = load_dataset(config.dataset, "bm-fr-en")
    common_voice['test'] = common_voice['test'].shuffle(seed=42).select(
        range(min(config.max_valid_size, len(common_voice['test']))))

    logger.info("Removing unnecessary columns and renaming 'english' to 'sentence'...")
    common_voice = common_voice.remove_columns(['french', 'duration', 'bambara']).rename_column("english", "sentence")

    # Set audio column to dataset format and resample if necessary
    logger.info(f"Resampling audio data to {config.feature_extractor.sampling_rate} Hz...")
    common_voice = common_voice.cast_column("audio", Audio(sampling_rate=config.feature_extractor.sampling_rate))

    # Apply dataset transformations: prepare and filter dataset
    logger.info("Preparing and filtering the dataset...")

    def prepare_batch(batch):
        """
        Preprocess batch of data.

        Args:
            batch (dict): A batch from the dataset.

        Returns:
            dict: A processed batch with audio features and labels.
        """
        # Load and possibly resample the audio data to the target sample rate
        audio = batch["audio"]
        feature_extractor = config.feature_extractor
        processor = config.processor
        normalizer = config.normalizer

        # Compute log-Mel spectrogram features from the audio signal
        input_features = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
        batch["input_features"] = input_features

        # Compute the input length in seconds
        batch["input_length"] = len(audio["array"]) / audio["sampling_rate"]

        # Optionally, preprocess the transcription
        transcription = batch["sentence"]
        if config.do_lower_case:
            transcription = transcription.lower()
        if config.do_remove_punctuation:
            transcription = normalizer(transcription).strip()

        # Encode the transcription into label ids
        batch["labels"] = processor.tokenizer(transcription).input_ids

        return batch

    common_voice = common_voice.map(
        prepare_batch,
        remove_columns=common_voice.column_names["train"]
    )

    common_voice["train"] = common_voice["train"].filter(
        lambda x: x < config.max_input_length,
        input_columns=["input_length"],
    )

    return common_voice
