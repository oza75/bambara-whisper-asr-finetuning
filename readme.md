# Bambara Whisper ASR Fine-Tuning

This repository contains scripts for fine-tuning the OpenAI Whisper model for automatic speech recognition (ASR) in
Bambara, though the approach can be adapted to other languages.

## Overview

The provided scripts enable users to fine-tune Whisper models on a specific language dataset, focusing on improving ASR
capabilities. This implementation leverages the Hugging Face `transformers` library and is designed to be flexible,
allowing modifications to fine-tune on other languages and datasets.

## Features

- CLI-based configuration for easy adjustment of training parameters.
- Integrated logging for both console and file output to track the training process.
- Modular design for easy customization and extension.

## Prerequisites

Before you begin, ensure you have the following installed:

- Python 3.10 or later

## Installation

First, clone this repository to your local machine:

```bash
git clone https://github.com/oza75/bambara-whisper-asr-finetuning.git
cd bambara-whisper-asr-finetuning
````

Then, install the required Python packages:

```bash
pip install --upgrade -r requirements.txt
```

## Usage

To start fine-tuning the Whisper model, use the `main.py` script. The script supports various command-line options to
customize the training parameters.

### Basic Usage

```
python main.py
```

### Advanced usage

You can specify training parameters using command-line arguments. Here are some examples:

```
accelerate launch --config-file accelerate_config.yaml main.py --deepspeed_config deepspeed_config.json --fp16 --model_checkpoint openai/whisper-medium --learning_rate 6.25e-06  --per_device_train_batch_size 64 --gradient_accumulation_steps 1 --per_device_eval_batch_size 32 --output_dir whisper-bambara-asr-002 --push_to_hub
```

### Available Command-Line Options

Available options for detailed tuning and configuration are listed in the config.py.

## Configuration

Modify the config.py to change default settings or add new parameters as needed for other languages or specific
requirements.

## Contributing

Contributions to this project are welcome! You can contribute in several ways:

- Submit bugs and feature requests.
- Review code and check for errors.
- Suggest enhancements to the documentation.

## License
This project is licensed under the MIT License.

## Acknowledgments
- OpenAI for the Whisper model.
- Hugging Face for the transformers library.