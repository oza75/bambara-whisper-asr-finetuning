import logging
import torch
import transformers.models.whisper.tokenization_whisper as whisper_tokenization
from tokenizers import AddedToken
from transformers import WhisperTokenizer, WhisperProcessor, WhisperForConditionalGeneration, WhisperTokenizerFast
from transformers.models.whisper.tokenization_whisper import TO_LANGUAGE_CODE, TASK_IDS
from dataclasses import dataclass
from typing import Any, Dict, List, Union, Tuple
from transformers import WhisperFeatureExtractor
from config import Configuration

logger = logging.getLogger(__name__)


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch


def setup_model_and_processor(config: Configuration) -> Tuple[
    WhisperForConditionalGeneration,
    WhisperProcessor,
    WhisperFeatureExtractor,
    WhisperTokenizerFast
]:
    """
    Set up the Whisper model and associated processor.

    Args:
        config (Configuration): The Configuration object containing all the settings.

    Returns:
        model (WhisperForConditionalGeneration): The instantiated and configured Whisper model.
        processor (WhisperProcessor): The instantiated and configured Whisper processor.
    """
    # Load the feature extractor
    logger.info(f"Loading feature extractor from model checkpoint '{config.model_checkpoint}'...")
    feature_extractor = WhisperFeatureExtractor.from_pretrained(config.model_checkpoint)

    # Load the tokenizer
    logger.info(f"Loading tokenizer for language '{config.language}'...")
    tokenizer = WhisperTokenizerFast.from_pretrained(
        config.model_checkpoint,
        language=config.language,
        task="transcribe"
    )

    # Create the processor using the feature extractor and tokenizer
    logger.info("Creating Whisper processor...")
    processor = WhisperProcessor(feature_extractor=feature_extractor, tokenizer=tokenizer)

    # Load the Whisper model
    logger.info(f"Loading model from checkpoint '{config.model_checkpoint}'...")
    model = WhisperForConditionalGeneration.from_pretrained(config.model_checkpoint)

    # Apply additional configuration to the model if necessary
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    model.config.use_cache = False

    # Set generation config parameters
    logger.info(f"Setting model generation config for language '{config.language}'...")
    model.generation_config.language = config.language.lower()
    model.generation_config.task = "transcribe"

    return model, processor, feature_extractor, tokenizer
