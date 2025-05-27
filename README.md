---
language: en
tags:
- audio
- text-to-speech
- english
- insurance
- sales
license: apache-2.0
datasets:
- custom
metrics:
- loss
---

# Insurance Sales Agent Voice Model

<p align="center">
<img src="assets/stree-banner.png" alt="Professional Insurance Sales Agent Voice Assistant" width="600"/>
</p>

<p align="center">
<a href="https://huggingface.co/prarabdha21/insurance-agent-voice"><img src="https://huggingface.co/datasets/huggingface/badges/resolve/main/model-on-hf-lg-dark.svg" alt="Model on HuggingFace" height=42></a>
</p>

Insurance Sales Agent Voice Model is a fine-tuned version of the Dia model, specifically optimized for generating professional insurance sales agent voice samples. The model has been trained on a diverse dataset of insurance-related conversations and can generate various types of professional interactions including policy explanations, customer consultations, and sales pitches.

## Model Description

- **Base Model:** Dia-1.6B
- **Fine-tuned for:** Professional Insurance Sales Voice Generation
- **Language:** English
- **License:** Apache 2.0

## Features

- Generates professional insurance sales voice samples
- Supports multiple interaction categories:
  - Policy explanations
  - Customer consultations
  - Sales pitches
  - Follow-up conversations
- Maintains high audio quality and professional tone
- Optimized for insurance industry voice generation

## Installation

```bash
pip install torch transformers numpy soundfile tqdm wandb huggingface_hub
```

## Usage

```python
from dia.model import Dia

# Load the model
model = Dia.from_pretrained("prarabdha21/insurance-agent-voice")

# Generate professional voice
text = "Let me explain the benefits of our comprehensive insurance policy..."
audio = model.generate(
    text=text,
    temperature=0.8,
    top_p=0.95,
    cfg_scale=2.0
)

# Save the generated audio
model.save_audio("output.wav", audio)
```

## Training Data

The model was fine-tuned on a custom dataset containing:
- Insurance policy explanations
- Sales conversation scripts
- Customer service interactions
- Professional consultation dialogues

## Hardware Requirements

- GPU with at least 10GB VRAM (recommended)
- CUDA 12.6 or later
- PyTorch 2.0 or later

## Limitations

- Best suited for insurance-related voice generation
- May not perform optimally for non-insurance content
- Requires appropriate text prompts for best results

## Citation

If you use this model in your research or project, please cite:

```bibtex
@misc{insurance-agent-voice,
  author = {Prarabdha},
  title = {Insurance Sales Agent Voice Model},
  year = {2024},
  publisher = {Hugging Face},
  journal = {Hugging Face Hub},
  howpublished = {\url{https://huggingface.co/prarabdha21/insurance-agent-voice}}
}
```

## License

This model is released under the Apache License 2.0. See the [LICENSE](LICENSE) file for more details.

## Disclaimer

This project offers a high-fidelity speech generation model intended for research and educational use. The following uses are **strictly forbidden**:

- **Identity Misuse**: Do not produce audio resembling real individuals without permission.
- **Deceptive Content**: Do not use this model to generate misleading content or false insurance claims
- **Illegal or Malicious Use**: Do not use this model for activities that are illegal or intended to cause harm.

By using this model, you agree to uphold relevant legal standards and ethical responsibilities. We **are not responsible** for any misuse and firmly oppose any unethical usage of this technology.
