import logging
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
from config import Configuration

logger = logging.getLogger(__name__)


def setup_training_args(config: Configuration):
    """
    Setup the training arguments for the Seq2SeqTrainer.

    Args:
        config (Configuration): The Configuration object containing all the settings.

    Returns:
        Seq2SeqTrainingArguments: Training arguments for the trainer.
    """
    logger.info("Setting up training arguments...")
    training_args = Seq2SeqTrainingArguments(
        torch_compile=True,
        output_dir=config.output_dir,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        warmup_steps=config.warmup_steps,
        optim=config.optim,
        weight_decay=config.weight_decay,
        gradient_checkpointing=config.gradient_checkpointing,
        fp16=config.fp16,
        lr_scheduler_type=config.lr_scheduler_type,
        evaluation_strategy=config.evaluation_strategy,
        save_strategy=config.save_strategy,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        predict_with_generate=config.predict_with_generate,
        generation_max_length=config.generation_max_length,
        save_steps=config.save_steps,
        eval_steps=config.eval_steps,
        logging_steps=config.logging_steps,
        report_to=config.report_to,
        accelerator_config=config.accelerate_config,
        deepspeed=config.deepspeed_config,
        load_best_model_at_end=config.load_best_model_at_end,
        metric_for_best_model=config.metric_for_best_model,
        greater_is_better=config.greater_is_better,
        push_to_hub=config.push_to_hub
    )
    return training_args


def run_training(trainer):
    """
    Run the training process.

    Args:
        trainer (Seq2SeqTrainer): Configured Seq2SeqTrainer instance.

    Returns:
        TrainerState: The state of the trainer after training.
    """
    return trainer.train()


def create_trainer(model, config: Configuration, data_collator, compute_metrics, train_dataset, eval_dataset):
    """
    Create a Seq2SeqTrainer instance.

    Args:
        model: The model to be trained.
        config (Configuration): Configuration object containing training parameters.
        data_collator: Data collator used for generating batches.
        compute_metrics (function): Function to compute metrics during evaluation.
        train_dataset (Dataset): The dataset for training.
        eval_dataset (Dataset): The dataset for evaluation.

    Returns:
        Seq2SeqTrainer: Configured trainer instance.
    """
    training_args = setup_training_args(config)
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics
    )
    return trainer
