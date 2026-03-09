import os
from typing import List, Tuple
import pandas as pd


def _get_slide_name(filename: str) -> str:
    return filename.split("_xmin")[0]


def load_splits(all_files: List[str], root: str, fold: int) -> Tuple[List[str], List[str]]:
    """ load train/test splits from CSV files 

    Args:
        all_files: list of all dataset filenames 
        root: dataset root directory 
        fold: fold number

    Returns:
        (train_files, val_files) — subsets of all_files.
    """
    splits_dir = os.path.join(root, "train_test_splits")
    train_csv = os.path.join(splits_dir, f"fold_{fold}_train.csv")
    val_csv = os.path.join(splits_dir, f"fold_{fold}_test.csv")

    if not os.path.exists(train_csv) or not os.path.exists(val_csv):
        raise FileNotFoundError(
            f"Split CSVs not found for fold {fold} in {splits_dir}. "            
        )

    train_slides = set(pd.read_csv(train_csv)["slide_name"].tolist())
    val_slides = set(pd.read_csv(val_csv)["slide_name"].tolist())

    train_files = [f for f in all_files if _get_slide_name(f) in train_slides]
    val_files = [f for f in all_files if _get_slide_name(f) in val_slides]

    unassigned = [f for f in all_files
                  if _get_slide_name(f) not in train_slides and _get_slide_name(f) not in val_slides]
    if unassigned:
        print(f"Warning: {len(unassigned)} files not found in split CSVs")

    return train_files, val_files
