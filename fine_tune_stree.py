import torch
from pathlib import Path
import json
import numpy as np
import soundfile as sf
from dia.model import Dia
import logging
import os
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import get_linear_schedule_with_warmup
import wandb

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('stree_fine_tuning.log'),
        logging.StreamHandler()
    ]
)

class StreeDataset(Dataset):
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

class StreeFineTuner:
    def __init__(self, base_model_path="nari-labs/Dia-1.6B", device="cuda"):
        logging.info(f"Initializing Stree-1.1A fine-tuner with device: {device}")
        self.device = torch.device(device)
        try:
            self.model = Dia.from_pretrained(base_model_path, compute_dtype="float16", device=self.device)
            # Get the underlying PyTorch model
            self.torch_model = self.model.model
            self.torch_model.train()  # Set to training mode
            logging.info("Base model loaded successfully")
        except Exception as e:
            logging.error(f"Error loading model: {e}")
            raise

    def train_step(self, texts, audio_prompts):
        """
        Perform a single training step
        """
        # Convert audio prompts to tensors if they aren't already
        audio_tensors = []
        for prompt in audio_prompts:
            if isinstance(prompt, np.ndarray):
                prompt = torch.from_numpy(prompt).to(self.device)
            audio_tensors.append(prompt)

        # Forward pass
        outputs = self.torch_model(
            texts=texts,
            audio_prompts=audio_tensors
        )
        
        # Calculate loss (you may need to adjust this based on your model's output)
        loss = outputs.loss
        
        return loss

    def prepare_dataset(self, data_dir):
        """
        Prepare dataset from multiple sources:
        - Emotional speech data
        - General speech data
        - Domain-specific data
        """
        logging.info(f"Loading data from {data_dir}")
        processed_data = []
        
        # Process all JSON files in the data directory
        for json_file in Path(data_dir).glob("*.json"):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                for item in tqdm(data['samples'], desc=f"Processing {json_file.name}"):
                    text = item['text']
                    # Update path to look in audio_files subdirectory
                    audio_path = Path(data_dir) / "audio_files" / item['audio_file']
                    
                    if audio_path.exists():
                        try:
                            audio_data, sr = sf.read(str(audio_path))
                            processed_data.append({
                                'text': text,
                                'audio': audio_data,
                                'sample_rate': sr,
                                'category': item.get('category', 'general')
                            })
                            logging.debug(f"Processed {audio_path.name}")
                        except Exception as e:
                            logging.warning(f"Error processing {audio_path.name}: {e}")
                    else:
                        logging.warning(f"Audio file not found: {audio_path}")
                        
            except Exception as e:
                logging.error(f"Error processing {json_file}: {e}")
                continue
        
        logging.info(f"Processed {len(processed_data)} audio files successfully")
        return processed_data

    def fine_tune(self, dataset, output_dir, num_epochs=10, batch_size=8, learning_rate=2e-5):
        """
        Fine-tune the model with improved training process
        """
        logging.info(f"Starting fine-tuning for {num_epochs} epochs")
        
        # Initialize wandb for experiment tracking
        wandb.init(project="stree-1.1a", name="fine-tuning")
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Create dataset and dataloader
        train_dataset = StreeDataset(dataset)
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4
        )
        
        # Training setup
        optimizer = torch.optim.AdamW(self.torch_model.parameters(), lr=learning_rate)
        total_steps = len(train_dataloader) * num_epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(0.1 * total_steps),
            num_training_steps=total_steps
        )
        
        best_loss = float('inf')
        
        
        for epoch in range(num_epochs):
            logging.info(f"Starting epoch {epoch+1}/{num_epochs}")
            total_loss = 0
            num_batches = 0
            
            for batch in tqdm(train_dataloader, desc=f"Epoch {epoch+1}"):
                try:
                    optimizer.zero_grad()
                    
                    # Forward pass
                    loss = self.train_step(
                        texts=[item['text'] for item in batch],
                        audio_prompts=[item['audio'] for item in batch]
                    )
                    
                    # Backward pass
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.torch_model.parameters(), max_norm=1.0)
                    optimizer.step()
                    scheduler.step()
                    
                    total_loss += loss.item()
                    num_batches += 1
                    
                    # Log metrics
                    wandb.log({
                        "loss": loss.item(),
                        "learning_rate": scheduler.get_last_lr()[0]
                    })
                    
                    if num_batches % 10 == 0:
                        logging.info(f"Batch {num_batches}, Loss: {loss.item():.4f}")
                        
                except Exception as e:
                    logging.error(f"Error in batch {num_batches}: {e}")
                    continue
            
            avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
            logging.info(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}")
            
            # Save checkpoint if it's the best so far
            if avg_loss < best_loss:
                best_loss = avg_loss
                try:
                    checkpoint_path = output_path / "best_model"
                    self.torch_model.save_pretrained(checkpoint_path)
                    logging.info(f"Saved best model checkpoint to {checkpoint_path}")
                except Exception as e:
                    logging.error(f"Error saving checkpoint: {e}")
            
            # Save regular checkpoint
            try:
                checkpoint_path = output_path / f"checkpoint_epoch_{epoch+1}"
                self.torch_model.save_pretrained(checkpoint_path)
                logging.info(f"Saved checkpoint to {checkpoint_path}")
            except Exception as e:
                logging.error(f"Error saving checkpoint: {e}")
        
        wandb.finish()

def main():
    try:
        # Initialize fine-tuner
        logging.info("Initializing Stree-1.1A fine-tuner")
        fine_tuner = StreeFineTuner()
        
        # Prepare dataset
        dataset = fine_tuner.prepare_dataset(
            data_dir="training_data"
        )
        
        if not dataset:
            logging.error("No valid data found for fine-tuning")
            return
        
        # Fine-tune model
        fine_tuner.fine_tune(
            dataset=dataset,
            output_dir="stree-1.1a-model",
            num_epochs=10,
            batch_size=8,
            learning_rate=2e-5
        )
        
        logging.info("Fine-tuning completed successfully")
        
    except Exception as e:
        logging.error(f"Error during fine-tuning: {e}")
        raise

if __name__ == "__main__":
    main() 