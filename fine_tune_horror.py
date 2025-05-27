import torch
from pathlib import Path
import json
import numpy as np
import soundfile as sf
from dia.model import Dia
import logging
import os
from tqdm import tqdm

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('fine_tuning.log'),
        logging.StreamHandler()
    ]
)

class HorrorHindiFineTuner:
    def __init__(self, base_model_path="nari-labs/Dia-1.6B", device="cuda"):
        logging.info(f"Initializing fine-tuner with device: {device}")
        self.device = torch.device(device)
        try:
            self.model = Dia.from_pretrained(base_model_path, compute_dtype="float16", device=self.device)
            logging.info("Model loaded successfully")
        except Exception as e:
            logging.error(f"Error loading model: {e}")
            raise
        
    def prepare_dataset(self, dialogues_file, audio_prompts_dir):
        """
        Prepare dataset from Hindi horror dialogues and audio prompts
        dialogues_file: JSON file with format:
        {
            "dialogues": [
                {
                    "text": "हा हा हा... मैं तुम्हारी आत्मा को खा जाऊंगा...",
                    "audio_prompt": "horror_laugh.wav"
                }
            ]
        }
        """
        logging.info(f"Loading dialogues from {dialogues_file}")
        try:
            with open(dialogues_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            logging.error(f"Error loading dialogues file: {e}")
            raise
        
        processed_data = []
        for item in tqdm(data['dialogues'], desc="Processing audio files"):
            text = item['text']
            audio_path = Path(audio_prompts_dir) / item['audio_prompt']
            
            if audio_path.exists():
                try:
                    audio_data, sr = sf.read(str(audio_path))
                    processed_data.append({
                        'text': text,
                        'audio': audio_data,
                        'sample_rate': sr
                    })
                    logging.debug(f"Processed {audio_path.name}")
                except Exception as e:
                    logging.warning(f"Error processing {audio_path.name}: {e}")
            else:
                logging.warning(f"Audio file not found: {audio_path}")
        
        logging.info(f"Processed {len(processed_data)} audio files successfully")
        return processed_data

    def fine_tune(self, dataset, output_dir, num_epochs=5, batch_size=4):
        """
        Fine-tune the model on horror Hindi dialogues
        """
        logging.info(f"Starting fine-tuning for {num_epochs} epochs")
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Training setup
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-5)
        
        for epoch in range(num_epochs):
            logging.info(f"Starting epoch {epoch+1}/{num_epochs}")
            total_loss = 0
            num_batches = 0
            
            for batch in tqdm(self._create_batches(dataset, batch_size), desc=f"Epoch {epoch+1}"):
                try:
                    optimizer.zero_grad()
                    
                    # Forward pass
                    loss = self.model.train_step(
                        texts=[item['text'] for item in batch],
                        audio_prompts=[item['audio'] for item in batch]
                    )
                    
                    # Backward pass
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item()
                    num_batches += 1
                    
                    if num_batches % 10 == 0:
                        logging.info(f"Batch {num_batches}, Loss: {loss.item():.4f}")
                        
                except Exception as e:
                    logging.error(f"Error in batch {num_batches}: {e}")
                    continue
            
            avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
            logging.info(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}")
            
            # Save checkpoint
            try:
                checkpoint_path = output_path / f"checkpoint_epoch_{epoch+1}"
                self.model.save_pretrained(checkpoint_path)
                logging.info(f"Saved checkpoint to {checkpoint_path}")
            except Exception as e:
                logging.error(f"Error saving checkpoint: {e}")
    
    def _create_batches(self, dataset, batch_size):
        """Create batches from dataset"""
        for i in range(0, len(dataset), batch_size):
            yield dataset[i:i + batch_size]

def main():
    try:
        # Initialize fine-tuner
        logging.info("Initializing fine-tuner")
        fine_tuner = HorrorHindiFineTuner()
        
        # Prepare dataset
        dataset = fine_tuner.prepare_dataset(
            dialogues_file="horror_dialogues.json",
            audio_prompts_dir="horror_audio_prompts"
        )
        
        if not dataset:
            logging.error("No valid data found for fine-tuning")
            return
        
        # Fine-tune model
        fine_tuner.fine_tune(
            dataset=dataset,
            output_dir="horror_hindi_model",
            num_epochs=5
        )
        
        logging.info("Fine-tuning completed successfully")
        
    except Exception as e:
        logging.error(f"Error during fine-tuning: {e}")
        raise

if __name__ == "__main__":
    main() 