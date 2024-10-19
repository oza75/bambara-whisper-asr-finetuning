from datasets import load_dataset, Audio, concatenate_datasets, Dataset
from config import Configuration
import logging
from typing import Dict

logger = logging.getLogger(__name__)


def duplicate_split(split: Dataset) -> Dict[str, Dataset]:
    """
    Duplicate the dataset for transcription and translation tasks.

    Args:
        split (Dataset): The dataset split to be duplicated.

    Returns:
        Dict[str, Dataset]: A dictionary containing both transcription and translation datasets.
    """
    transcription_dataset = split.remove_columns(['french', 'duration', 'english'])
    transcription_dataset = transcription_dataset.add_column(
        "task_type", ["transcribe"] * len(split)
    ).rename_column("bambara", "sentence")

    translation_dataset = split.remove_columns(['french', 'duration', 'bambara'])
    translation_dataset = translation_dataset.add_column(
        "task_type", ["translate"] * len(split)
    ).rename_column("english", "sentence")

    return {
        "transcribe": transcription_dataset,
        "translate": translation_dataset
    }


def load_and_prepare_dataset(config: Configuration, already_preprocessed: bool = True) -> Dict[str, Dataset]:
    """
    Load and prepare the Whisper dataset for training.

    Args:
        config (Configuration): The Configuration object containing all the settings.
        already_preprocessed (bool): Whether to load the pre-processed dataset.
    Returns:
        Dict[str, Dataset]: A dictionary containing the processed 'train' and 'test' datasets.
    """
    logger.info(f"Loading the {config.dataset} dataset from the Hugging Face Hub...")
    common_voice = load_dataset(config.dataset, "bm-fr-en")

    if already_preprocessed:
        return common_voice

    # Set audio column to the dataset format and resample if necessary for all splits
    logger.info(f"Resampling audio data to {config.feature_extractor.sampling_rate} Hz for all splits...")
    common_voice = common_voice.cast_column("audio", Audio(sampling_rate=config.feature_extractor.sampling_rate))

    # Shuffle and limit the size of the validation set to ensure efficient evaluation
    logger.info("Shuffling and limiting the size of the test set...")
    common_voice['test'] = common_voice['test'].shuffle(seed=42).select(
        range(min(config.max_valid_size, len(common_voice['test'])))
    )

    # Duplicate the dataset for transcription and translation tasks for each split
    logger.info("Duplicating dataset for transcription and translation tasks...")
    train_splits = duplicate_split(common_voice['train'])
    test_splits = duplicate_split(common_voice['test'])

    # Prepare and process each split with the appropriate task type
    logger.info("Preparing and processing transcription and translation datasets...")

    def prepare_batch(batch: Dict, task_type: str) -> Dict:
        """
        Preprocess batch of data.

        Args:
            batch (dict): A batch from the dataset.
            task_type (str): The type of task, either 'transcribe' or 'translate'.

        Returns:
            dict: A processed batch with audio features and labels.
        """
        # Load and resample the audio data to the target sample rate
        audio = batch["audio"]
        feature_extractor = config.feature_extractor
        processor = config.processor
        normalizer = config.normalizer

        # Compute log-Mel spectrogram features from the audio signal
        input_features = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
        batch["input_features"] = input_features

        # Compute the input length in seconds for filtering
        batch["input_length"] = len(audio["array"]) / audio["sampling_rate"]

        # Preprocess the transcription or translation text
        sentences = batch["sentence"]
        if config.do_lower_case:
            sentences = sentences.lower()
        if config.do_remove_punctuation:
            sentences = normalizer(sentences).strip()

        # Set prefix tokens based on the task type
        processor.tokenizer.set_prefix_tokens(language=config.language.lower(), task=task_type)

        # Encode the transcription into label ids
        batch["labels"] = processor.tokenizer(sentences, truncation=True, max_length=448).input_ids

        return batch

    # Apply the batch preparation function to both transcription and translation datasets using a loop
    processed_splits = {}
    for split_name, split_data in {"train": train_splits, "test": test_splits}.items():
        processed_splits[split_name] = []
        for task_type, dataset in split_data.items():
            processed_dataset = dataset.map(
                lambda batch: prepare_batch(batch, task_type=task_type),
                remove_columns=dataset.column_names,
                cache_file_name=f"./caches/{split_name}_{task_type}_prepared.arrow"
            )
            processed_splits[split_name].append(processed_dataset)

    # Concatenate the transcription and translation datasets and shuffle
    logger.info("Concatenating and shuffling the datasets...")
    common_voice['train'] = concatenate_datasets(processed_splits['train']).shuffle(seed=42)
    common_voice['test'] = concatenate_datasets(processed_splits['test']).shuffle(seed=42)

    # Filter out audio samples that are too long for the model to process efficiently
    logger.info("Filtering out samples exceeding the maximum input length...")
    common_voice['train'] = common_voice['train'].filter(
        lambda x: x < config.max_input_length,
        input_columns=["input_length"],
    )
    common_voice['test'] = common_voice['test'].filter(
        lambda x: x < config.max_input_length,
        input_columns=["input_length"],
    )

    return common_voice
