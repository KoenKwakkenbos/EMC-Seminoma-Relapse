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

from models import create_classification_model
from helpers import tar_path, dir_path

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


def dir_path(path):
    """
    Check if a path is a directory.

    Parameters:
    - path (str): The path to check.

    Returns:
    - str: The validated path if it is a directory.

    Raises:
        NotADirectoryError: If the path is not a directory.
    """
    if os.path.isdir(path):
        return path
    raise NotADirectoryError(path)


def tar_path(path):
    """
    Check if a path is a directory. If it is not, create the directory and return the path.

    Parameters:
    - path (str): The path to check.

    Returns:
    - str: The validated path as a directory.

    """
    if not os.path.isdir(path):
        os.makedirs(path)
        print(f"Output folder {path} created")
        return path
    return path

def compute_class_weights(labels):
    labels = np.array(labels)

    negative_count = len(np.where(labels==0)[0])
    positive_count = len(np.where(labels==1)[0])
    total_count = negative_count + positive_count

    return {
        0: (1 / negative_count) * (total_count / 2),
        1: (1 / positive_count) * (total_count / 2),
    }


def train(train_gen, val_gen, weights, out_path, fold):
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

    with strategy.scope():
        model = create_classification_model((224, 224, 3))
        model.compile(
            optimizer="adam", loss='binary_crossentropy', metrics=['accuracy', tf.keras.metrics.AUC()]
        )

    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=40,
        class_weight=weights,
        callbacks=[early_stopping, model_checkpoint, reduce_lr],
        verbose=1,
    )

    # Load best weights.
    model.load_weights(file_path)
    
    return model


def main():
    # Get input from commandline
    parsed_args = parse_arguments()

    patient_data = pd.read_csv('./Seminoma_Outcomes_Anon.csv', header=0).set_index('AnonPID')
    
    skf = StratifiedKFold(n_splits=3)
    skf.get_n_splits(patient_data.index, patient_data['Meta'])

    for i, (train_index, test_index) in enumerate(skf.split(patient_data.index, patient_data['Meta'])):
        K.clear_session()
        print(f"Fold {i}:")
        print(f"  Train: index={train_index}")
        print(f"  Test:  index={test_index}")

        print(patient_data.iloc[train_index])

        train_gen = Datagen(list(patient_data.iloc[train_index].index), patient_data['Meta'].iloc[train_index], 224, parsed_args.input, batch_size=32, train=True, imagenet=True)
        val_gen = Datagen(list(patient_data.iloc[test_index].index), patient_data['Meta'].iloc[test_index], 224, parsed_args.input, batch_size=32, train=False, imagenet=True)

        train_weights = compute_class_weights(patient_data['Meta'].iloc[train_index])

        model = train(train_gen, val_gen, train_weights, parsed_args.output, i)


if __name__ == "__main__":
    main()
