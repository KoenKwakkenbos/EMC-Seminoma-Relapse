"""
Script used to run the MIL experiment.

The file can be run with two arguments (input and output). The input
path should be the folder containing the tiles. The output is optional
but will be the folder where the models will be written to. The output
path defaults to ./ (the current working directory).
The MIL training routine required a custom training loop, so that the
top-k tiles can be set in each epoch. A clear documentation can be found on:
-> https://keras.io/guides/writing_a_training_loop_from_scratch/
The functions used below are adapted from this tutorial. 

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
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
import tensorflow.keras.backend as K

from datagenerator import MILdatagen
from models import create_MIL_model
from helpers import dir_path, tar_path

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


# The three following @tf functions are taken from the following tutorial:
# https://keras.io/guides/writing_a_training_loop_from_scratch/.

@tf.function
def train_step(x, y):
    with tf.GradientTape() as tape:

        # Run the forward pass of the layer.
        # The operations that the layer applies
        # to its inputs are going to be recorded
        # on the GradientTape.
        logits = model(x, training=True)  # Logits for this minibatch

        # Compute the loss value for this minibatch.
        loss_value = loss_fn(y, logits)
    # Use the gradient tape to automatically retrieve
    # the gradients of the trainable variables with respect to the loss.
    grads = tape.gradient(loss_value, model.trainable_weights)

    # Run one step of gradient descent by updating
    # the value of the variables to minimize the loss.
    optimizer.apply_gradients(zip(grads, model.trainable_weights))
    train_acc_metric.update_state(y, logits)

    return loss_value

@tf.function
def inference_step(x):
    # Only pass the samples through and collect the output vales
    return model(x, training=False)

@tf.function
def test_step(x, y):
    val_logits = model(x, training=False)
    val_acc_metric.update_state(y, val_logits)
    val_auc_metric.update_state(y, tf.nn.sigmoid(val_logits))
    loss_value = loss_fn(y, val_logits)
    return loss_value

def group_argtopk(groups, data, k=10):
    """
    Ranks the tiles per slide in decending order, keeping the top k.
    Code mimimally adapted from: 
    https://github.com/MSKCC-Computational-Pathology/MIL-nature-medicine-2019/blob/master/MIL_train.py

    Parameters:
    - groups: A list or array specifying the slide each tile originates from.
    - data: The model's predictions per tile.
    - k (int, optional): The number of tiles to keep for training per slide.

    Returns:
    - List: The indices of all the top ranking tiles.
    """
    data = data.numpy().ravel()
    order = np.lexsort((data, groups))
    groups = np.array(groups)
    groups = groups[order]
    data = data[order]
    index = np.empty(len(groups), 'bool')
    index[-k:] = True
    index[:-k] = groups[k:] != groups[:-k]
    return list(order[index])


    

if __name__ == "__main__":
    parsed_args = parsed_args = parse_arguments()
    patient_data = pd.read_csv('./Seminoma_Outcomes_Anon.csv',
                                header=0).set_index('AnonPID')

    # Three-fold cross validation
    skf = StratifiedKFold(n_splits=3)
    skf.get_n_splits(patient_data.index, patient_data['Meta'])

    for i, (train_index, test_index) in enumerate(skf.split(patient_data.index,
                                                    patient_data['Meta'])):
        
        # New model gets instantiated, so clear cache
        K.clear_session()
        print(f"Fold {i}:")
        print(f"  Train: index={train_index}")
        print(f"  Test:  index={test_index}")

        print(patient_data.iloc[train_index])

        train_gen = MILdatagen(list(patient_data.iloc[train_index].index), patient_data['Meta'].iloc[train_index], 224, parsed_args.input, batch_size=32, train=True)
        val_gen = MILdatagen(list(patient_data.iloc[test_index].index), patient_data['Meta'].iloc[test_index], 224, parsed_args.input, batch_size=32, train=False)

        model = create_MIL_model(input_shape=(224, 224, 3))

        # Instantiate an optimizer.
        optimizer = keras.optimizers.Adam(lr=0.0001)
        # Instantiate a loss function.
        loss_fn = keras.losses.BinaryCrossentropy(from_logits=True)

        # Instantiate the metrics
        train_acc_metric = keras.metrics.BinaryAccuracy()
        val_acc_metric = keras.metrics.BinaryAccuracy()
        val_auc_metric = tf.keras.metrics.AUC()

        epochs = 40
        # Variables used to record the best loss and AUC (set to +/- inf. so that the first value is always recorded)
        best_val_loss = np.Inf
        best_val_auc = -np.Inf

        for epoch in range(epochs):
            # Reset loss values after each epoch:
            avg_loss = 0
            avg_loss_val = 0

            print("\nStart of epoch %d" % (epoch,))

            # Iterate over the batches of the dataset.
            # Collect the logits from the inference pass of all batches
            logits = tf.zeros([0, 1])
            for step, (x_batch_train, y_batch_train) in enumerate(tqdm(train_gen)):
                logits_batch = inference_step(x_batch_train)
                logits = tf.concat([logits, logits_batch], 0)

            # Get the slides used in the training generator, so that the group_argtopk function can
            # rank the tiles on slide level.
            slides = train_gen.slide_tile_list[:min(len(train_gen.slide_tile_list), logits.shape[0])]
            topk_idx = group_argtopk(slides, logits, k=10)

            train_gen.topk_dataset(topk_idx)
            for step, (x_batch_train, y_batch_train) in enumerate(tqdm(train_gen)):
                loss_value = train_step(x_batch_train, y_batch_train)
                avg_loss += loss_value

            # Divide the accumulated loss by the number of steps taken so that it is averaged.
            avg_loss /= (step+1)

            # Print training results
            train_acc = train_acc_metric.result()
            print("Training acc over epoch: %.4f" % (float(train_acc),))
            print(
                "Training loss (for one batch) at step %d: %.4f"
                % (step, float(avg_loss))
            )

            # Validation
            logits = tf.zeros([0, 1])
            for step, (x_batch_val, y_batch_val) in enumerate(tqdm(val_gen)):
                logits_batch = inference_step(x_batch_val)
                logits = tf.concat([logits, logits_batch], 0)

            slides = val_gen.slide_tile_list[:min(len(val_gen.slide_tile_list), logits.shape[0])]
            topk_idx = group_argtopk(slides, logits, k=10)

            val_gen.topk_dataset(topk_idx)

            for step, (x_batch_val, y_batch_val) in enumerate(tqdm(val_gen)):
                loss_value = test_step(x_batch_val, y_batch_val)
                avg_loss_val += loss_value
            avg_loss_val /= (step+1)

            # Save model if the loss is lower than the best loss at that point.
            if avg_loss_val < best_val_loss:
                model.save_weights(os.path.join(parsed_args.output, f"./output_lr/best_model_weights_MILsmall_loss_fold{i+1}.h5"))
                best_val_loss = avg_loss_val


            # Print validation results
            val_acc = val_acc_metric.result()
            val_auc = val_auc_metric.result()
            #if val_auc > best_val_auc:
            #    model.save_weights(f"./output_lr/best_model_weights_MILsmall_auc_fold{i+1}.h5")
            #    best_val_auc = val_auc

            print("Validation loss: %.4f" % (float(avg_loss_val),))
            print("Validation acc: %.4f" % (float(val_acc),))
            print("Validation auc: %.4f" % (float(val_auc),))

            # Reset metrics and generators for the next epoch
            train_acc_metric.reset_states()
            val_acc_metric.reset_states()
            val_auc_metric.reset_states()
            train_gen.on_epoch_end()
            val_gen.on_epoch_end()

        print(f'----------------------------- Fold {i+1} completed -----------------------------')