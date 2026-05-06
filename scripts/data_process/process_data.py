import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_from_disk, DatasetDict, Dataset
from sklearn.model_selection import train_test_split

from scripts.utils.io_utils import ensure_dir, validate_paths
from scripts.utils.logging_config import get_logger
from scripts.utils.constants import LABEL_MAP, TEXT_COLUMN, LABEL_COLUMN

logger = get_logger(__name__, log_file="outputs/logs/data_process.log")

# Default paths
DEFAULT_RAW_DIR = "data/raw"
DEFAULT_OUTPUT_SPLIT_DIR = "data/processed/split"
DEFAULT_EDA_DIR = "outputs/plots"


def process_banking77(
    raw_dir: str = DEFAULT_RAW_DIR,
    output_split_dir: str = DEFAULT_OUTPUT_SPLIT_DIR,
    eda_dir: str = DEFAULT_EDA_DIR,
    random_state: int = 42,
    val_split_ratio: float = 0.1
):
    """Process Banking77 dataset: deduplicate, map labels, split train/val/test, and generate EDA visualizations.
    
    Args:
        raw_dir: Directory containing raw dataset from load_data.py
        output_split_dir: Directory to save processed splits
        eda_dir: Directory to save EDA visualizations
        random_state: Random seed for reproducibility
        val_split_ratio: Validation split ratio (default 0.1 = 90% train, 10% val)
    """
    logger.info("Starting Banking77 dataset processing...")
    
    # Validate input paths
    try:
        validate_paths([raw_dir])
    except FileNotFoundError as e:
        logger.error(f"Input path validation failed: {e}")
        raise
    
    # 1. Load raw dataset
    try:
        logger.info(f"Loading raw dataset from: {raw_dir}")
        ds = load_from_disk(raw_dir)
        logger.info(f"Loaded dataset with splits: {list(ds.keys())}")
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        raise
    
    # 2. Deduplicate Train split
    logger.info("Deduplicating training data...")
    train_df = ds["train"].to_pandas()
    original_len = len(train_df)
    train_df = train_df.drop_duplicates(subset=[TEXT_COLUMN], keep="first")
    logger.info(f"Removed {original_len - len(train_df)} duplicate rows")
    
    # 3. Map labels (int -> snake_case str) using LABEL_MAP from constants
    logger.info("Mapping label integers to intent strings...")
    train_df[LABEL_COLUMN] = train_df[LABEL_COLUMN].map(LABEL_MAP)
    test_df = ds["test"].to_pandas()
    test_df[LABEL_COLUMN] = test_df[LABEL_COLUMN].map(LABEL_MAP)
    logger.info(f"Mapped labels. Unique labels: {train_df[LABEL_COLUMN].nunique()}")
    
    # 4. EDA: Label Distribution
    logger.info("Generating EDA visualizations...")
    eda_path = ensure_dir(eda_dir)
    for name, df in [("train", train_df), ("test", test_df)]:
        try:
            plt.figure(figsize=(10, 15))
            sns.countplot(data=df, y=LABEL_COLUMN, order=df[LABEL_COLUMN].value_counts().index)
            plt.title(f"Label Distribution: {name.upper()}")
            plt.tight_layout()
            save_path = eda_path / f"{name}_dist.png"
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
            plt.close()
            logger.info(f"Saved visualization: {save_path}")
        except Exception as e:
            logger.warning(f"Failed to save EDA plot for {name}: {e}")
    
    # 5. Stratified Split (Train -> Train/Val with specified ratio)
    logger.info(f"Splitting training data: {1-val_split_ratio:.1%} train, {val_split_ratio:.1%} validation")
    train_final, val_final = train_test_split(
        train_df,
        test_size=val_split_ratio,
        stratify=train_df[LABEL_COLUMN],
        random_state=random_state
    )
    logger.info(f"Split sizes - Train: {len(train_final)}, Val: {len(val_final)}, Test: {len(test_df)}")
    
    # 6. Convert back to HF Dataset and save
    logger.info("Converting to HuggingFace Dataset format...")
    processed_ds = DatasetDict({
        "train": Dataset.from_pandas(train_final.reset_index(drop=True)),
        "val": Dataset.from_pandas(val_final.reset_index(drop=True)),
        "test": Dataset.from_pandas(test_df.reset_index(drop=True))
    })
    
    output_path = ensure_dir(output_split_dir)
    try:
        processed_ds.save_to_disk(str(output_path))
        logger.info(f"✓ Processing complete. Dataset saved to: {output_path}")
    except Exception as e:
        logger.error(f"Failed to save processed dataset: {e}")
        raise


if __name__ == "__main__":
    process_banking77()