cohort_settings = {
    'data_path': './tiles-normalized',
    'cohort_file': 'W:/train_val_cohort.xlsx',
    'synchronous': 0,
    'treatment': 1,
    'magnification': 10
}

model_settings = {
        "dropout": 0.5,
        "optimizer": "Adam",
        "loss": "CPHloss",
        "metric": "CIndex",
        "tile_size": 512,
        "clinical_vars": [],
}

train_settings = {
    "learning_rate": 0.001,
    "epochs": 100,
    "batch_size": 64
}