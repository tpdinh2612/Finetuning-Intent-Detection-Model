from datasets import load_from_disk
from transformers import AutoTokenizer

from scripts.utils.io_utils import ensure_dir, validate_paths
from scripts.utils.logging_config import get_logger
from scripts.utils.constants import SYSTEM_PROMPT, TEXT_COLUMN, LABEL_COLUMN

logger = get_logger(__name__, log_file="outputs/logs/data_process.log")

# Default paths and model
DEFAULT_SPLIT_DIR = "data/processed/split"
DEFAULT_TEMPLATE_DIR = "data/processed/template"
DEFAULT_MODEL_ID = "unsloth/Llama-3.2-1B-Instruct"


def apply_llama_template(
    split_dir: str = DEFAULT_SPLIT_DIR,
    save_dir: str = DEFAULT_TEMPLATE_DIR,
    model_id: str = DEFAULT_MODEL_ID,
    show_sample: bool = True
):
    """Apply Llama chat template to dataset splits for SFT training.
    
    Converts original (text, label) format into chat template prompts
    suitable for supervised fine-tuning with transformers SFTTrainer.
    
    Args:
        split_dir: Directory containing train/val/test splits from process_banking77()
        save_dir: Directory to save formatted dataset
        model_id: HuggingFace model ID to use for chat template
        show_sample: Whether to log a sample formatted prompt
    """
    logger.info("Starting dataset formatting with chat template...")
    
    # Validate input
    try:
        validate_paths([split_dir])
    except FileNotFoundError as e:
        logger.error(f"Input path validation failed: {e}")
        raise
    
    # Load splits
    try:
        logger.info(f"Loading dataset splits from: {split_dir}")
        ds = load_from_disk(split_dir)
        logger.info(f"Loaded splits: {list(ds.keys())} with {sum(len(ds[k]) for k in ds)} total examples")
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        raise
    
    # Load tokenizer
    try:
        logger.info(f"Loading tokenizer from: {model_id}")
        tokenizer = AutoTokenizer.from_pretrained(model_id)
    except Exception as e:
        logger.error(f"Failed to load tokenizer: {e}")
        raise
    
    def format_row(example):
        """Convert (text, label) to chat template prompt."""
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Classify: {example[TEXT_COLUMN]}"},
            {"role": "assistant", "content": example[LABEL_COLUMN]}
        ]
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )
        return {TEXT_COLUMN: prompt}
    
    # Format all splits
    logger.info("Applying chat template to all splits...")
    try:
        formatted_ds = ds.map(
            format_row,
            remove_columns=ds["train"].column_names,
            desc="Formatting"
        )
        logger.info(f"Formatted dataset with {sum(len(formatted_ds[k]) for k in formatted_ds)} examples")
    except Exception as e:
        logger.error(f"Failed to format dataset: {e}")
        raise
    
    # Save formatted dataset
    output_path = ensure_dir(save_dir)
    try:
        formatted_ds.save_to_disk(str(output_path))
        logger.info(f"✓ Formatted dataset saved to: {output_path}")
    except Exception as e:
        logger.error(f"Failed to save formatted dataset: {e}")
        raise
    
    # Show sample prompt if requested
    if show_sample:
        sample_prompt = formatted_ds["train"][0][TEXT_COLUMN]
        logger.info(f"\n--- Sample formatted prompt (first 1000 chars) ---\n{sample_prompt[:1000]}...")


if __name__ == "__main__":
    apply_llama_template()