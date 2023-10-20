cohort_settings = {
    'data_path': './tiles-normalized',
    'cohort_file': 'W:/train_val_cohort.xlsx',
    'synchronous': 0,
    'treatment': 0,
    'magnification': 10
}

model_settings = {
        "dropout": 0.2,
        "optimizer": "Adam",
        "loss": "CPHloss",
        "metric": "CIndex",
        "tile_size": 512
}

train_settings = {
    "learning_rate": 0.001,
    "epochs": 100,
    "batch_size": 32
}