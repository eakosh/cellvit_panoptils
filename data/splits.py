# Copyright © 2025 Olena Kosharova, FIIT STU.
# Part of the "Tissue-Context CellViT Extension" (bachelor's thesis, FIIT STU).
# Licensed under the Apache License 2.0 with the Commons Clause restriction.
# See the LICENSE file in the project root for full terms.

import os
from typing import List, Tuple
import pandas as pd


def _get_slide_name(filename: str) -> str:
    return filename.split("_xmin")[0]


def load_splits(all_files: List[str], root: str, fold: int,) -> Tuple[List[str], List[str]]:

    splits_dir = os.path.join(root, "5_kfold_splits")
    train_csv = os.path.join(splits_dir, f"fold_{fold}_train.csv")
    val_csv = os.path.join(splits_dir, f"fold_{fold}_val.csv")

    if not os.path.exists(train_csv) or not os.path.exists(val_csv):
        raise FileNotFoundError(f"Split CSVs not found for fold {fold} in {splits_dir}")

    train_slides = set(pd.read_csv(train_csv)["slide_name"].tolist())
    val_slides = set(pd.read_csv(val_csv)["slide_name"].tolist())

    train_files = [f for f in all_files if _get_slide_name(f) in train_slides]
    val_files = [f for f in all_files if _get_slide_name(f) in val_slides]

    return train_files, val_files


def load_dev_split(all_files: List[str], root: str, num_val_hospitals: int = 10) -> Tuple[List[str], List[str]]:
    splits_dir = os.path.join(root, "5_kfold_splits")
    test_csv = os.path.join(splits_dir, "test.csv")
    if not os.path.exists(test_csv):
        raise FileNotFoundError(f"test.csv not found in {splits_dir}")

    test_slides = set(pd.read_csv(test_csv)["slide_name"].tolist())

    frames = []
    for fold in range(1, 6):
        for part in ("train", "val"):
            p = os.path.join(splits_dir, f"fold_{fold}_{part}.csv")
            if os.path.exists(p):
                frames.append(pd.read_csv(p))
    if not frames:
        raise FileNotFoundError(f"No fold CSVs found in {splits_dir}")

    dev_df = (pd.concat(frames).drop_duplicates("slide_name").reset_index(drop=True))
    dev_df = dev_df[~dev_df["slide_name"].isin(test_slides)]

    foldable = dev_df[dev_df["has_nuclei"].fillna(True).astype(bool)]
    hospital_sizes = foldable.groupby("hospital").size().sort_values()
    val_hospitals = set(hospital_sizes.index[:num_val_hospitals])

    val_slides = set(foldable[foldable["hospital"].isin(val_hospitals)]["slide_name"])
    train_slides = set(dev_df[~dev_df["hospital"].isin(val_hospitals)]["slide_name"])

    train_files = [f for f in all_files if _get_slide_name(f) in train_slides]
    val_files = [f for f in all_files if _get_slide_name(f) in val_slides]

    return train_files, val_files


def load_test_split(all_files: List[str], root: str) -> List[str]:

    test_csv = os.path.join(root, "5_kfold_splits", "test.csv")

    if not os.path.exists(test_csv):
        raise FileNotFoundError(f"test.csv not found in {os.path.dirname(test_csv)}")

    test_slides = set(pd.read_csv(test_csv)["slide_name"].tolist())
    
    return [f for f in all_files if _get_slide_name(f) in test_slides]
