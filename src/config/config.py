# -*- coding: utf-8 -*-
"""Model config in json format"""

CFG = {
    "data": {
        "raw": "../data/raw/CAMUS_public/database_nifti/",
        "processed": "../data/processed/",
    },
    "train": {
        "seed": 42,
        "batch_size": 64,
        "epoch": 10,
        "train_portion": 0.7,
        "val_portion": 0.2,
        "lr": 1e-3
    },
    "model": {
        "input_size": 10,
        "hidden_size": 256,
        "num_layers": 3,
        "dropout": 0.25,
        "n_classes": 10,
        "save_to": "saved/"
    }
}
