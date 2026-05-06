from datasets import load_dataset

from scripts.utils.io_utils import ensure_dir, validate_paths
from scripts.utils.logging_config import get_logger


logger = get_logger(__name__, log_file="outputs/logs/data_process.log")

# Default path for raw dataset
DEFAULT_RAW_DATA_DIR = "data/raw"


def download_banking77(save_dir: str = DEFAULT_RAW_DATA_DIR):
    """Download Banking77 dataset from HuggingFace and save to local disk.
    
    Args:
        save_dir: Directory to save the dataset. Defaults to 'data/raw'.
        
    Returns:
        Downloaded dataset object from HuggingFace.
        
    Raises:
        Exception: If dataset download or save fails.
    """
    logger.info("Downloading Banking77 dataset from HuggingFace...")
    
    try:
        dataset = load_dataset("banking77")
        logger.info(f"Successfully loaded dataset with {len(dataset)} splits")
    except Exception as e:
        logger.error(f"Failed to download dataset: {e}")
        raise
    
    # Ensure output directory exists
    output_dir = ensure_dir(save_dir)
    logger.info(f"Output directory prepared: {output_dir}")
    
    # Save dataset to disk
    try:
        dataset.save_to_disk(str(output_dir))
        logger.info(f"Dataset saved to: {output_dir}")
    except Exception as e:
        logger.error(f"Failed to save dataset: {e}")
        raise
    
    return dataset


if __name__ == "__main__":
    download_banking77()