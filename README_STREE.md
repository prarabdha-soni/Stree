# Stree-1.1A: Fine-tuned Speech Synthesis Model

Stree-1.1A is a fine-tuned version of the Dia model, optimized for high-quality speech synthesis with improved emotional expressiveness and natural prosody.

## Features

- Enhanced emotional speech synthesis
- Improved natural prosody and intonation
- Better handling of various speaking styles
- Optimized for both general and emotional speech

## Setup

1. Install the required dependencies:
```bash
pip install torch transformers soundfile numpy tqdm wandb
```

2. Create a training data directory structure:
```
training_data/
├── emotional_speech.json
├── general_speech.json
└── audio_files/
    ├── happy_1.wav
    ├── sad_1.wav
    └── ...
```

3. Prepare your training data:
   - Place your audio files in the `training_data/audio_files/` directory
   - Update the JSON files with your text-audio pairs
   - Ensure audio files are in WAV format with appropriate sample rate

## Fine-tuning

To fine-tune the model:

```bash
python fine_tune_stree.py
```

The script will:
1. Load the base Dia model
2. Process your training data
3. Fine-tune the model with the specified parameters
4. Save checkpoints and the best model

## Training Parameters

You can modify the following parameters in `fine_tune_stree.py`:

- `num_epochs`: Number of training epochs (default: 10)
- `batch_size`: Batch size for training (default: 8)
- `learning_rate`: Learning rate for optimization (default: 2e-5)

## Uploading to Hugging Face

After fine-tuning, to upload your model to Hugging Face:

1. Install the Hugging Face Hub library:
```bash
pip install huggingface_hub
```

2. Login to Hugging Face:
```bash
huggingface-cli login
```

3. Create a new model repository on Hugging Face

4. Upload your model:
```bash
python -c "
from huggingface_hub import HfApi
api = HfApi()
api.upload_folder(
    folder_path='stree-1.1a-model',
    repo_id='your-username/stree-1.1a',
    repo_type='model'
)
"
```

## Model Usage

After uploading to Hugging Face, you can use the model:

```python
from dia.model import Dia

# Load the model
model = Dia.from_pretrained("your-username/stree-1.1a")

# Generate speech
audio = model.generate("Your text here")
```

## Training Data Format

The training data should be organized in JSON files with the following structure:

```json
{
    "samples": [
        {
            "text": "Your text here",
            "audio_file": "audio_file.wav",
            "category": "category_name"
        }
    ]
}
```

## Contributing

Feel free to contribute to the project by:
1. Adding more training data
2. Improving the fine-tuning process
3. Reporting issues and suggesting improvements

## License

This project is licensed under the same license as the base Dia model. 