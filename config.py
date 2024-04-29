import argparse


class Configuration:
    """
    Configuration settings for the Whisper ASR model training.

    This class is responsible for handling training configurations,
    which can be overridden via command-line arguments.
    """

    def __init__(self, args):
        """
        Initializes the configuration with values from command-line arguments or defaults.

        Args:
            args (argparse.Namespace): Parsed command-line arguments.
        """
        self.max_valid_size = args.max_valid_size
        self.model_checkpoint = args.model_checkpoint
        self.output_dir = args.output_dir
        self.language = args.language
        self.do_lower_case = args.do_lower_case
        self.do_remove_punctuation = args.do_remove_punctuation
        self.max_input_length = args.max_input_length
        self.do_normalize_eval = args.do_normalize_eval
        self.num_train_epochs = args.num_train_epochs
        self.per_device_train_batch_size = args.per_device_train_batch_size
        self.gradient_accumulation_steps = args.gradient_accumulation_steps
        self.learning_rate = args.learning_rate
        self.lr_scheduler_type = args.lr_scheduler_type
        self.warmup_steps = args.warmup_steps
        self.gradient_checkpointing = args.gradient_checkpointing
        self.fp16 = args.fp16
        self.evaluation_strategy = args.evaluation_strategy
        self.save_strategy = args.save_strategy
        self.per_device_eval_batch_size = args.per_device_eval_batch_size
        self.predict_with_generate = args.predict_with_generate
        self.generation_max_length = args.generation_max_length
        self.save_steps = args.save_steps
        self.eval_steps = args.eval_steps
        self.logging_steps = args.logging_steps
        self.report_to = args.report_to
        self.accelerate_config = args.accelerate_config
        self.deepspeed_config = args.deepspeed_config
        self.load_best_model_at_end = args.load_best_model_at_end
        self.metric_for_best_model = args.metric_for_best_model
        self.greater_is_better = args.greater_is_better
        self.push_to_hub = args.push_to_hub
        self.dataset = args.dataset
        # Need to be initialized after the model definition
        self.feature_extractor = None
        self.normalizer = None
        self.processor = None
        self.tokenizer = None

    @classmethod
    def from_argparse_args(cls, args):
        """
        Creates a Configuration instance from argparse Namespace.

        Args:
            args (argparse.Namespace): Parsed command-line arguments.

        Returns:
            Configuration: An instance of the Configuration class.
        """
        return cls(args)


def parse_args():
    """
    Parses command-line arguments for the Whisper ASR model training script.

    Returns:
        argparse.Namespace: The namespace containing all the parsed CLI arguments.
    """
    parser = argparse.ArgumentParser(description="Fine-tuning the Whisper ASR model.")

    # Dataset and model parameters
    parser.add_argument('--max_valid_size', type=int, default=700,
                        help='Maximum size of the validation dataset.')
    parser.add_argument('--model_checkpoint', type=str, default="oza75/whisper-bambara-asr-001",
                        help='Pretrained model checkpoint for initialization.')
    parser.add_argument('--language', type=str, default="bambara",
                        help='Language code for the ASR model.')
    parser.add_argument('--do_lower_case', action='store_true', default=False,
                        help='Convert all transcriptions to lowercase.')
    parser.add_argument('--do_remove_punctuation', action='store_true', default=False,
                        help='Remove punctuation from transcriptions.')
    parser.add_argument('--do_normalize_eval', action='store_true', default=True,
                        help="Evaluate with the 'normalised' WER.")
    parser.add_argument('--max_input_length', type=float, default=30.0,
                        help='Maximum length of audio input in seconds for filtering during training.')

    parser.add_argument('--dataset', type=str, default="oza75/bambara-asr", help='The dataset to use for training.')

    # Training hyperparameters
    parser.add_argument('--num_train_epochs', type=int, default=10,
                        help='Total number of training epochs.')
    parser.add_argument('--per_device_train_batch_size', type=int, default=8,
                        help='Batch size per GPU/TPU core/CPU for training.')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=8,
                        help='Number of updates steps to accumulate before performing a backward/update pass.')
    parser.add_argument('--learning_rate', type=float, default=1e-5,
                        help='Initial learning rate.')
    parser.add_argument('--lr_scheduler_type', type=str, default='linear',
                        help='The learning rate scheduler type, default to linear.')
    parser.add_argument('--warmup_steps', type=int, default=500,
                        help='Number of warmup steps for learning rate scheduler.')
    parser.add_argument('--accelerate_config', type=str, default=None, help='Path to the Accelerate configuration file')
    parser.add_argument('--deepspeed_config', type=str, default=None, help='Path to the DeepSpeed configuration file')

    # Training arguments specific to Huggingface Transformers
    parser.add_argument('--gradient_checkpointing', action='store_true', default=True,
                        help='Use gradient checkpointing to save memory at the cost of slower backward pass.')
    parser.add_argument('--fp16', action='store_true', default=True,
                        help='Whether to use 16-bit (mixed) precision instead of 32-bit.')
    parser.add_argument('--evaluation_strategy', type=str, default="steps",
                        help='The evaluation strategy to use.')
    parser.add_argument('--save_strategy', type=str, default="steps",
                        help='The checkpoint save strategy to use.')
    parser.add_argument('--per_device_eval_batch_size', type=int, default=16,
                        help='Batch size per GPU/TPU core/CPU for evaluation.')
    parser.add_argument('--predict_with_generate', action='store_true', default=True,
                        help='Whether to use generate to calculate generative metrics (ROUGE, BLEU).')
    parser.add_argument('--generation_max_length', type=int, default=240,
                        help='The maximum length of the sequence to be generated during evaluation.')
    parser.add_argument('--save_steps', type=int, default=50,
                        help='Number of training steps between model checkpoint saves.')
    parser.add_argument('--eval_steps', type=int, default=50,
                        help='Number of training steps between evaluations.')
    parser.add_argument('--logging_steps', type=int, default=25,
                        help='Number of training steps between logging events.')

    # Output and push to hub
    parser.add_argument('--output_dir', type=str, default='whisper-bambara-asr-001',
                        help='The output directory where the model predictions and checkpoints will be written.')
    parser.add_argument('--load_best_model_at_end', action='store_true', default=True,
                        help='Whether to load the best model found at each evaluation.')
    parser.add_argument('--metric_for_best_model', type=str, default="wer",
                        help='The metric to use to compare model performance.')
    parser.add_argument('--greater_is_better', action='store_true', default=False,
                        help='Whether a larger value of the metric indicates a better model.')
    parser.add_argument('--push_to_hub', action='store_true',
                        help='Whether to push the model to the Hugging Face model hub after training.')

    # Report to Hugging Face Hub
    parser.add_argument('--report_to', type=str, nargs='+', default=["tensorboard"],
                        help='The list of integrations to report the logs to.')

    # Parse and return the arguments
    return parser.parse_args()
