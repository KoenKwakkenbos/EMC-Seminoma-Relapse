import os
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tf_dataset_binary import get_datagenerator
from sklearn.model_selection import train_test_split, StratifiedKFold
import tensorflow.keras.backend as K
import wandb


from binary_models import *
import config_resnet_10 as config
from pathlib import Path
from tqdm import tqdm
import numpy as np

from wandb.keras import WandbMetricsLogger

print('starting')

def extract_identifier(filename):
    # Attempt extraction using both hyphen and underscore separators
    separators = ['-', '_']
    for separator in separators:
        parts = filename.split(separator)
        if len(parts) >= 3:
            identifier = separator.join(parts[:3])
            return identifier
    return None


def train_val_split(tile_list, df_train, df_val, clinical_vars):
    tile_list_train = [file for file in tile_list
                       if extract_identifier(os.path.basename(file)) in df_train.index]
    tile_list_val = [file for file in tile_list
                     if extract_identifier(os.path.basename(file)) in df_val.index]
    
    # Perform oversampling
    labels = df_train.loc[[extract_identifier(os.path.basename(file)) for file in tile_list_train]]['Event'].to_list()
    class_counts = {label: labels.count(label) for label in set(labels)}
    print(f"Training --> oversampling. Class distribution before oversampling: {class_counts}")

    majority_class = 0
    minority_class = 1

    num_oversample_majority = int(class_counts[majority_class] * 0.20)
    num_oversample_minority = num_oversample_majority + (class_counts[majority_class] - class_counts[minority_class])

    majority_images = [img for img, label in zip(tile_list_train, labels) if label == majority_class]
    minority_images = [img for img, label in zip(tile_list_train, labels) if label == minority_class]

    oversampled_majority = np.random.choice(majority_images, num_oversample_majority, replace=True)
    oversampled_minority = np.random.choice(minority_images, num_oversample_minority, replace=True)

    tile_list_train.extend(oversampled_majority.tolist())
    tile_list_train.extend(oversampled_minority.tolist())

    class_counts[majority_class] += num_oversample_majority
    class_counts[minority_class] += num_oversample_minority
    print(f"Class distribution after oversampling: {class_counts}")

    train_ids = [extract_identifier(os.path.basename(file)) for file in tile_list_train]
    labels_train = df_train.loc[[extract_identifier(os.path.basename(file)) for file in tile_list_train]]['Event'].to_list()
    clin_train = np.array(df_train.loc[train_ids][clinical_vars], dtype=np.float32)
    val_ids = [extract_identifier(os.path.basename(file)) for file in tile_list_val]
    labels_val = df_val.loc[[extract_identifier(os.path.basename(file)) for file in tile_list_val]]['Event'].to_list()
    clin_val = np.array(df_val.loc[val_ids][clinical_vars], dtype=np.float32)
    return tile_list_train, clin_train, labels_train, tile_list_val, clin_val, labels_val


class TrainAndEvaluateModel:
    def __init__(self, model, model_dir, train_dataset, eval_dataset,
                 learning_rate, num_epochs):
        self.num_epochs = num_epochs
        self.model_dir = model_dir

        self.model = model

        self.train_ds = train_dataset
        self.val_ds = eval_dataset

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.loss_fn = 'binary_crossentropy'
   

    def train_and_evaluate(self):
        file_path = os.path.join(self.model_dir, f"best_model_weights.h5")

        # Initialize model checkpoint callback.
        model_checkpoint = keras.callbacks.ModelCheckpoint(
            file_path,
            monitor="val_loss",
            verbose=0,
            mode="min",
            save_best_only=True,
            save_weights_only=True,
        )

        # Initialize early stopping callback.
        # The model performance is monitored across the validation data and stops training
        # when the generalization error cease to decrease.
        early_stopping = keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=15, mode="min"
        )
        
        reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1,
                            patience=7, min_lr=0.000001)

        # Compile model.
        model = self.model
        model.compile(
            optimizer=self.optimizer, loss=self.loss_fn, metrics=['accuracy', tf.keras.metrics.AUC(), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
        )

        model.fit(
            self.train_ds,
            validation_data=self.val_ds,
            epochs=self.num_epochs,
            callbacks=[early_stopping, model_checkpoint, reduce_lr, WandbMetricsLogger()],
            verbose=1
        )

# STRATIFIED 5-FOLD CROSS VALIDATION
patient_data = pd.read_excel(config.cohort_settings['cohort_file'], header=0, engine="openpyxl").set_index('ID').dropna()

if not config.cohort_settings['synchronous']:
    patient_data = patient_data.drop(patient_data[patient_data['Synchronous'] == 1].index)
if not config.cohort_settings['treatment']:
    patient_data = patient_data.drop(patient_data[patient_data['Treatment'] == 1].index)

skf = StratifiedKFold(n_splits=5)
skf.get_n_splits(patient_data.index, patient_data['Event'])


for i, (train_index, val_index) in enumerate(skf.split(patient_data.index, patient_data['Event'])):
    # Start a run, tracking hyperparameters
    wandb.init(
        # set the wandb project where this run will be logged
        project="ResNet_Fully_Supervised_Binary",

        # track hyperparameters and run metadata with wandb.config
        config={
            'Fold': i+1,
            'Model': 'ResNet50',
            **config.cohort_settings,
            **config.model_settings,
            **config.train_settings
        },
        reinit=True
    )

    K.clear_session()
    print(f"Fold {i+1}:")
    print(f"  Train: index={train_index}")
    print(f"  Test:  index={val_index}")

    patient_data_train = patient_data.iloc[train_index].copy()
    patient_data_val = patient_data.iloc[val_index].copy()

    clinical_vars = ['RTI', 'LVI', 'Size']

    tile_list = [os.path.join(subdir, file) 
                 for subdir, dirs, files in os.walk(config.cohort_settings['data_path'])
                 for file in files if file.lower().endswith(('.png', '.jpg'))]

    tiles_train, clin_train, labels_train, tiles_val, clin_val, labels_val = train_val_split(tile_list, patient_data_train, patient_data_val, config.model_settings['clinical_vars'])

    train_gen = get_datagenerator(tiles_train, clin_train, labels_train, config.train_settings['batch_size'], train=True, imagenet=True)
    val_gen = get_datagenerator(tiles_val, clin_val, labels_val, config.train_settings['batch_size'], train=False, imagenet=True)

    model = create_imagenet_model(input_shape=(config.model_settings['tile_size'], config.model_settings['tile_size'], 3), num_clinical_features=0)
    print(model.summary())

    trainer = TrainAndEvaluateModel(
        model=model,
        model_dir=Path(f"ResNet_Fully_Supervised_Binaryl_fold_{i+1}"),
        train_dataset=train_gen,
        eval_dataset=val_gen,
        learning_rate=config.train_settings['learning_rate'],
        num_epochs=config.train_settings['epochs'],
    )

    trainer.train_and_evaluate()

    print('done')
    # wandb.join()
