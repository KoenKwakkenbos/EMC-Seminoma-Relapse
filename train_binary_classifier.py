"""
Script used to train the ResNet-50 classifier on the seminoma tiles.

The file can be run with two arguments (input and output). The input
path should be the folder containing the tiles. The output is optional
but will be the folder where the models will be written to. The output
path defaults to ./ (the current working directory).

Author: Koen Kwakkenbos
(k.kwakkenbos@student.tudelft.nl/k.kwakkenbos@gmail.com)
Version: 1.0
Date: Feb 2022
"""

import os
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from datagenerator import Datagen
from sklearn.model_selection import train_test_split, StratifiedKFold
import tensorflow.keras.backend as K

from models import create_imagenet_model
from helpers import tar_path, dir_path
from tf_dataset_binary import get_datagenerator
import config_resnet_10 as config

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


def parse_arguments():
    """
    Parse command line arguments.

    Returns:
    - argparse.Namespace: A namespace containing the parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Process commandline arguments.")

    parser.add_argument('-i', '--input', type=dir_path,
                        help="Path to the stored tiles.",
                        required=True)
    parser.add_argument('-o', '--output', type=tar_path, nargs="?", const=1,
                        help="Path to output folder.",
                        default='./')

    return parser.parse_args()


def compute_class_weights(labels):
    labels = np.array(labels)

    negative_count = len(np.where(labels==0)[0])
    positive_count = len(np.where(labels==1)[0])
    total_count = negative_count + positive_count

    return {
        0: (1 / negative_count) * (total_count / 2),
        1: (1 / positive_count) * (total_count / 2),
    }


def train(train_gen, val_gen, out_path, fold):
    file_path = os.path.join(out_path, f"best_model_weights_fold{fold+1}.h5")

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
        monitor="val_loss", patience=10, mode="min"
    )
    
    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1,
						  patience=5, min_lr=0.000001)

    # Compile model.
    strategy = tf.distribute.MirroredStrategy()
    print('Number of devices {}'.format(strategy.num_replicas_in_sync))

    model = create_imagenet_model((224, 224, 3))
    model.compile(
        optimizer="adam", loss='binary_crossentropy', metrics=['accuracy', tf.keras.metrics.AUC()]
    )

    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=40,
        callbacks=[early_stopping, model_checkpoint, reduce_lr],
        verbose=1,
    )

    # Load best weights.
    model.load_weights(file_path)
    
    return model


def main():
    # Get input from commandline
    parsed_args = parse_arguments()

    # patient_data = pd.read_csv('./Seminoma_Outcomes_Anon.csv', header=0).set_index('AnonPID')
    patient_data = pd.read_excel(config.cohort_settings['cohort_file'], header=0, engine="openpyxl").set_index('ID')

    if not config.cohort_settings['synchronous']:
        patient_data = patient_data.drop(patient_data[patient_data['Synchronous'] == 1].index)
    # if not config.cohort_settings['treatment']:
        # patient_data = patient_data.drop(patient_data[patient_data['Treatment'] == 1].index)

    print(patient_data.head(10))

    skf = StratifiedKFold(n_splits=3)
    skf.get_n_splits(patient_data.index, patient_data['Event'])

    for i, (train_index, val_index) in enumerate(skf.split(patient_data.index, patient_data['Event'])):
        K.clear_session()
        print(f"Fold {i}:")
        print(f"  Train: index={train_index}")
        print(f"  Test:  index={val_index}")

        print(patient_data.iloc[train_index])
        
        patient_data_train = patient_data.iloc[train_index].copy()
        patient_data_val = patient_data.iloc[val_index].copy()

        clinical_vars = ['RTI', 'LVI', 'Size']

        tile_list = [os.path.join(subdir, file) 
                    for subdir, dirs, files in os.walk(config.cohort_settings['data_path'])
                    for file in files if file.lower().endswith(('.png', '.jpg'))]

        tiles_train, clin_train, labels_train, tiles_val, clin_val, labels_val = train_val_split(tile_list, patient_data_train, patient_data_val, config.model_settings['clinical_vars'])

        train_gen = get_datagenerator(tiles_train, clin_train, labels_train, config.train_settings['batch_size'], train=True, imagenet=True)
        val_gen = get_datagenerator(tiles_val, clin_val, labels_val, config.train_settings['batch_size'], train=False, imagenet=True)

        model = train(train_gen, val_gen, parsed_args.output, i)


if __name__ == "__main__":
    main()