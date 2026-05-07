"""Generate 5-fold splits for PanopTILs

Strategy (uses original dataset authors' split via hospital logic):

- reserve test hospitals: A7, EW, GI (~15% of slides)
These hospitals are used only for the test set because the model should not see 
specifics of equipment beforehand

- remaining 17 hospitals split to 5-fold hospital-stratified CV
Hospitals assigned to folds with greed balance on descending size so 
each fold has a similar number of slides and their hospitals do not intersect 

- slides without nuclei are excluded from val and test

Outputs:
    test.csv
    fold_{N}_train.csv
    fold_{N}_val.csv
"""

import os
import random
import pandas as pd
from collections import defaultdict

SEED = 42
TEST_HOSPITALS = {"A7", "EW", "GI"}     # hospitals only for test


def main():
    root = os.path.join(
        "PanopTILs_data",
        "BootstrapNucleiManualRegions_TCGA_05132021",
    )
    splits_dir = os.path.join(root, "5_kfold_splits")

    # collect all unique slides from the original CSVs
    orig_splits_dir = os.path.join(root, "train_test_splits")
    frames = []
    for fold in range(1, 6):
        path_train = os.path.join(orig_splits_dir, f"fold_{fold}_train.csv")
        path_test = os.path.join(orig_splits_dir, f"fold_{fold}_test.csv")
        frames.append(pd.read_csv(path_train))
        frames.append(pd.read_csv(path_test))

    all_slides = pd.concat(frames).drop_duplicates("slide_name").reset_index(drop=True)
    print(f"Total unique slides in original CSVs: {len(all_slides)}")

    # split into temp(train)/test
    test_mask = all_slides["hospital"].isin(TEST_HOSPITALS)
    test_df   = all_slides[test_mask].copy()
    tmp_df    = all_slides[~test_mask].copy()

    # remove slides without nuclei labels from val and test
    train_only_mask = ~tmp_df["has_nuclei"].fillna(True).astype(bool)
    train_only_df   = tmp_df[train_only_mask].copy()
    with_labels_df     = tmp_df[~train_only_mask].copy()

    print(f"Test set:\t{len(test_df)} slides")
    print(f"Train total:\t{len(tmp_df)} slides")
    print(f"\tWith labels:\t{len(with_labels_df)} slides")
    print(f"\tTrain-only:\t{len(train_only_df)} slides")

    # split hospitals to 5 folds
    hospital_sizes = with_labels_df.groupby("hospital").size().sort_values(ascending=False)

    fold_hospitals: dict[int, list[str]] = defaultdict(list)
    fold_counts = [0] * 5 

    for hospital, size in hospital_sizes.items():
        target_fold = fold_counts.index(min(fold_counts))
        fold_hospitals[target_fold + 1].append(hospital)
        fold_counts[target_fold] += size

    print("\nFold hospital split:")
    for fold_id in range(1, 6):
        hospitals = fold_hospitals[fold_id]
        n = with_labels_df[with_labels_df["hospital"].isin(hospitals)].shape[0]
        print(f"\tfold {fold_id}: {hospitals} ({n} slides)")


    test_path = os.path.join(splits_dir, "test.csv")
    test_df.to_csv(test_path, index=False)
    print(f"\ntest.csv saved: {test_path}")

    for fold_id in range(1, 6):
        val_hospitals = set(fold_hospitals[fold_id])
        val_df   = with_labels_df[with_labels_df["hospital"].isin(val_hospitals)].copy()
        train_df = pd.concat([
            with_labels_df[~with_labels_df["hospital"].isin(val_hospitals)],
            train_only_df[~train_only_df["hospital"].isin(val_hospitals)],
        ]).copy()

        val_path = os.path.join(splits_dir, f"fold_{fold_id}_val.csv")
        train_path = os.path.join(splits_dir, f"fold_{fold_id}_train.csv")

        val_df.to_csv(val_path, index=False)
        train_df.to_csv(train_path, index=False)
        print(f"Saved: fold {fold_id} train={len(train_df)} val={len(val_df)}")

    print("\nComplete")


if __name__ == "__main__":
    random.seed(SEED)
    main()
