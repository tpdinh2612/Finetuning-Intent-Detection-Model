# =========================================================
# Model Merge & Push to Hugging Face Hub
# =========================================================
"""
Merge DoRA adapter into base Llama-3.2-1B-Instruct model.
Save locally and push to Hugging Face Hub for sharing.
"""

import os
from pathlib import Path
from typing import Tuple

import dotenv
import torch
from huggingface_hub import login
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from scripts.utils.io_utils import ensure_dir, validate_paths
from scripts.utils.logging_config import get_logger

logger = get_logger(__name__, log_file="outputs/logs/merge.log")

# =========================================================
# CONFIGURATION
# =========================================================
dotenv.load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
HF_USERNAME = os.getenv("HF_USERNAME")
REPO_NAME = "Llama-3.2-1B-Banking77-Intent-Classification"

# Paths
ADAPTER_PATH = "models/checkpoints/finetuned_dora"
LOCAL_SAVE_PATH = "models/finetuned/llama-3.2-1b-banking77-merged"

# Model constants
BASE_MODEL_ID = "unsloth/Llama-3.2-1B-Instruct"


def validate_config() -> None:
    """Validate that required credentials and paths exist.
    
    Raises:
        ValueError: If HF_TOKEN or HF_USERNAME are not set
        FileNotFoundError: If adapter checkpoint directory doesn't exist
    """
    if not HF_TOKEN:
        raise ValueError("HF_TOKEN environment variable not set. Add to .env file.")
    if not HF_USERNAME:
        raise ValueError("HF_USERNAME environment variable not set. Add to .env file.")
    
    try:
        validate_paths([ADAPTER_PATH])
        logger.info("✓ All paths validated successfully")
    except FileNotFoundError as e:
        logger.error(f"Path validation failed: {e}")
        raise


def login_huggingface() -> None:
    """Authenticate with Hugging Face Hub using stored token."""
    try:
        logger.info("Logging in to Hugging Face Hub...")
        login(token=HF_TOKEN, add_to_git_credential=True)
        logger.info("✓ Hugging Face authentication successful")
    except Exception as e:
        logger.error(f"Failed to authenticate with Hugging Face: {e}")
        raise


def load_base_model() -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load base Llama model and tokenizer.
    
    Returns:
        Tuple of (model, tokenizer)
    """
    logger.info(f"Loading base model: {BASE_MODEL_ID}")
    
    try:
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_ID,
            torch_dtype=torch.float16,
            device_map="cpu",
        )
        logger.info(f"✓ Base model loaded")
    except Exception as e:
        logger.error(f"Failed to load base model: {e}")
        raise
    
    return base_model


def load_adapter_and_merge(base_model: AutoModelForCausalLM) -> AutoModelForCausalLM:
    """Load DoRA adapter and merge into base model.
    
    Args:
        base_model: The base Llama model
        
    Returns:
        Merged model with adapters integrated
    """
    logger.info(f"Loading DoRA adapter from: {ADAPTER_PATH}")
    
    try:
        adapter_model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
        logger.info("✓ Adapter loaded")
    except Exception as e:
        logger.error(f"Failed to load adapter: {e}")
        raise
    
    logger.info("Merging adapter weights into base model (this may take 15-30 seconds)...")
    
    try:
        merged_model = adapter_model.merge_and_unload()
        logger.info("✓ Adapter merged successfully")
    except Exception as e:
        logger.error(f"Failed to merge adapter: {e}")
        raise
    
    return merged_model


def save_local(merged_model: AutoModelForCausalLM) -> None:
    """Save merged model and tokenizer to local disk.
    
    Args:
        merged_model: The merged model to save
    """
    output_path = ensure_dir(LOCAL_SAVE_PATH)
    logger.info(f"Saving merged model to: {output_path}")
    
    try:
        merged_model.save_pretrained(str(output_path), safe_serialization=True)
        logger.info("✓ Model saved")
    except Exception as e:
        logger.error(f"Failed to save model: {e}")
        raise
    
    # Save tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH)
        tokenizer.save_pretrained(str(output_path))
        logger.info("✓ Tokenizer saved")
    except Exception as e:
        logger.error(f"Failed to save tokenizer: {e}")
        raise


def push_to_hub(merged_model: AutoModelForCausalLM) -> None:
    """Push merged model and tokenizer to Hugging Face Hub.
    
    Args:
        merged_model: The merged model to push
    """
    hf_repo_id = f"{HF_USERNAME}/{REPO_NAME}"
    logger.info(f"Pushing model to Hugging Face Hub: {hf_repo_id}")
    logger.info("This may take several minutes depending on internet speed...")
    
    try:
        merged_model.push_to_hub(hf_repo_id, safe_serialization=True)
        logger.info("✓ Model pushed")
    except Exception as e:
        logger.error(f"Failed to push model: {e}")
        raise
    
    # Push tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH)
        tokenizer.push_to_hub(hf_repo_id)
        logger.info("✓ Tokenizer pushed")
    except Exception as e:
        logger.error(f"Failed to push tokenizer: {e}")
        raise
    
    # Final message
    logger.info("=" * 70)
    logger.info("🎉 SUCCESS! Model published to Hugging Face Hub!")
    logger.info(f"📍 View at: https://huggingface.co/{hf_repo_id}")
    logger.info("=" * 70)


def main() -> None:
    """Main orchestration: validate → load → merge → save → push."""
    try:
        # 1. Validate
        validate_config()
        
        # 2. Login
        login_huggingface()
        
        # 3. Load & merge
        base_model = load_base_model()
        merged_model = load_adapter_and_merge(base_model)
        
        # 4. Save locally
        save_local(merged_model)
        
        # 5. Push to Hub
        push_to_hub(merged_model)
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise


if __name__ == "__main__":
    main()