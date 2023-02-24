"""
Script used to train the convolutional autoencoders (CAE).

The same model is used in both experiments. As the input is 224x224x3 regardless of the original regions, the experiments
for 512 and 1024-sized regions can be initiated using the command line argument
to the correct folder with the tiles.

Author: Koen Kwakkenbos
(k.kwakkenbos@student.tudelft.nl/k.kwakkenbos@gmail.com)
Version: 1.0
Date: Feb 2022
"""
# Imports
import os
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split

from datagenerator import AEDatagen
from models import create_autoencoder_model
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

def SSIMLoss(y_true, y_pred):
    """
    Custom loss function for the CAE.
    
    Parameters:
    - y_true (np.array): The original image.
    - y_pred (np.array): The reconstructed image.

    Returns:
    - The weighed SSIM loss.
    """
    return 0.5*(1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, 1.0))) + 0.5*keras.losses.mean_absolute_error(y_true, y_pred) + 0.5*keras.losses.mean_squared_error(y_true, y_pred)


def train(train_gen, val_gen, out_path):
    """
    Function used in training the CAE.
    
    Parameters:
    - train_gen: Datagenerator for training phase.
    - val_gen: Datagenerator for validation phase.

    Returns:
    - Trained CAE model (with weights corresponding to the best validation loss
      observed during training).
    """
    # Define path for saving best model
    file_path = os.path.join(out_path, "best_model_weights.h5")

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
    
    # Reduce the lr when no improvement in validation loss is observed.
    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1,
						  patience=5, min_lr=0.000001)
    
    # Compile model, using multiple GPUs (available on BIGR cluster)
    strategy = tf.distribute.MirroredStrategy()
    print('Number of devices {}'.format(strategy.num_replicas_in_sync))

    with strategy.scope():
        model = create_autoencoder_model((224, 224, 3))
        model.compile(
            optimizer="adam", loss=[SSIMLoss], run_eagerly=True
        )
        print(model.summary())

    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=200,
        callbacks=[early_stopping, model_checkpoint, reduce_lr],
        verbose=1,
    )

    # Load best weights.
    model.load_weights(file_path)
    
    return model


def main():
    parsed_args = parse_arguments()
    patient_data = pd.read_csv('./Seminoma_Outcomes_Anon.csv', header=0).set_index('AnonPID')
    print(patient_data.head())
   
    pat_train, pat_val, y_train, y_val = train_test_split(
        patient_data.index, patient_data['Meta'], test_size=0.25, random_state=42,
        stratify=patient_data['Meta']
    )

    print(list(pat_train))

    # Initialize datagenerators
    train_gen = AEDatagen(list(pat_train), y_train, 224, parsed_args.input, batch_size=64, train=True)
    val_gen = AEDatagen(list(pat_val), y_val, 224, parsed_args.input, batch_size=64, train=False)

    model = train(train_gen, val_gen, parsed_args.output)

if __name__ == "__main__":
    main()
