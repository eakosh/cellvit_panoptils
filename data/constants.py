# Copyright © 2025 Olena Kosharova, FIIT STU.
# Part of the "Tissue-Context CellViT Extension" (bachelor's thesis, FIIT STU).
# Licensed under the Apache License 2.0 with the Commons Clause restriction.
# See the LICENSE file in the project root for full terms.

NUCLEI_SUPERCLASS_MAP = {
    "epithelium": [1, 6],
    "stroma":     [2, 3],
    "tils":       [4, 5],
    "other":      [7],
}

TISSUE_CLASS_NAMES = {
    1: "epithelium",
    2: "stroma",
    3: "tils_dense",
    4: "other",
}

TISSUE_REMAP = {
    0: 0,  # Exclude 
    1: 1,  # Tumor to Epithelium
    2: 2,  # Stroma
    3: 3,  # TILs-dense
    4: 1,  # Normal epithelium to Epithelium
    5: 4,  # Junk/Debris to Other
    6: 4,  # Blood to Other
    7: 4,  # Other
    8: 4,  # Empty/Background → Other
}

NUCLEI_TISSUE_COMPATIBILITY = {
    1: [1],
    2: [2, 3],
    3: [2, 3],
    4: [2, 3],
    5: [2, 3],
    6: [1],
    7: [4],
    9: [1, 2, 3, 4],
}
