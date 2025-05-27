import os
from huggingface_hub import snapshot_download
from pathlib import Path

def download_model():
    # Create models directory if it doesn't exist
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    print("Downloading model from Hugging Face Hub...")
    try:
        # Download the model and its configuration
        model_path = snapshot_download(
            repo_id="nari-labs/Dia-1.6B",
            local_dir=models_dir / "Dia-1.6B",
            local_dir_use_symlinks=False
        )
        print(f"Model downloaded successfully to: {model_path}")
        return model_path
    except Exception as e:
        print(f"Error downloading model: {e}")
        raise

if __name__ == "__main__":
    download_model() 