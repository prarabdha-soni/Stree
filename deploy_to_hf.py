import os
from pathlib import Path
import logging
from huggingface_hub import HfApi, create_repo
from transformers import AutoModelForCausalLM, AutoTokenizer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('deployment.log'),
        logging.StreamHandler()
    ]
)

def deploy_to_huggingface(
    model_path: str,
    repo_name: str,
    repo_type: str = "model",
    private: bool = False,
    token: str = None
):
    """
    Deploy the fine-tuned model to Hugging Face Hub
    
    Args:
        model_path: Path to the fine-tuned model
        repo_name: Name of the repository on Hugging Face Hub
        repo_type: Type of repository (model, dataset, space)
        private: Whether the repository should be private
        token: Hugging Face API token
    """
    try:
        # Initialize Hugging Face API
        api = HfApi(token=token)
        
        # Create repository if it doesn't exist
        try:
            create_repo(
                repo_id=repo_name,
                repo_type=repo_type,
                private=private,
                token=token
            )
            logging.info(f"Created repository: {repo_name}")
        except Exception as e:
            logging.info(f"Repository {repo_name} already exists or error: {e}")
        
        # Upload model files
        logging.info(f"Uploading model from {model_path} to {repo_name}")
        api.upload_folder(
            folder_path=model_path,
            repo_id=repo_name,
            repo_type=repo_type
        )
        
        logging.info(f"Successfully deployed model to {repo_name}")
        
    except Exception as e:
        logging.error(f"Error deploying model: {e}")
        raise

def main():
    # Get Hugging Face token from environment variable
    token = os.getenv("HUGGINGFACE_TOKEN")
    if not token:
        raise ValueError("Please set HUGGINGFACE_TOKEN environment variable")
    
    # Model deployment configuration
    model_path = "stree-1.1a-model"  # Path to your fine-tuned model
    repo_name = "prarabdha21/stree-1.1a"  # Replace with your Hugging Face username
    
    # Deploy model
    deploy_to_huggingface(
        model_path=model_path,
        repo_name=repo_name,
        token=token
    )

if __name__ == "__main__":
    main() 