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
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
import tensorflow.keras.backend as K
import wandb

from datagenerator_survival import DatagenMILSurv
from models import create_MIL_model, create_classification_model
from helpers import dir_path, tar_path
from survival_model import CoxPHLoss, CindexMetric
from pathlib import Path

import config

print('starting')
# Start a run, tracking hyperparameters
wandb.init(
    # set the wandb project where this run will be logged
    project="survival-mil-resnet",

    # track hyperparameters and run metadata with wandb.config
    config={
        "layer_1": 512,
        "activation_1": "relu",
        # "dropout": random.uniform(0.01, 0.80),
        "layer_2": 10,
        "activation_2": "softmax",
        "optimizer": "sgd",
        "loss": "sparse_categorical_crossentropy",
        "metric": "accuracy",
        "epoch": 8,
        "batch_size": 256
    }
)

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


class TrainAndEvaluateModel:
    def __init__(self, model, model_dir, train_dataset, eval_dataset, learning_rate, num_epochs):
        self.num_epochs = num_epochs

        self.model = model
        self.model_dir = model_dir

        self.train_ds = train_dataset
        self.val_ds = eval_dataset

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.loss_fn = CoxPHLoss()

        self.train_loss_metric = tf.keras.metrics.Mean(name="train_loss")
        self.train_cindex_metric = CindexMetric()
        self.val_loss_metric = tf.keras.metrics.Mean(name="val_loss")
        self.val_cindex_metric = CindexMetric()

    @tf.function
    def train_one_step(self, x, y_event, y_riskset):
        y_event = tf.expand_dims(y_event, axis=1)
        with tf.GradientTape() as tape:

            # Run the forward pass of the layer.
            # The operations that the layer applies
            # to its inputs are going to be recorded
            # on the GradientTape.
            logits = model(x, training=True)  # Logits for this minibatch

            # Compute the loss value for this minibatch.
            train_loss = self.loss_fn(y_true=[y_event, y_riskset], y_pred=logits)
        # Use the gradient tape to automatically retrieve
        # the gradients of the trainable variables with respect to the loss.
        with tf.name_scope("gradients"):
            grads = tape.gradient(train_loss, self.model.trainable_weights)
            self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
        # train_acc_metric.update_state(y, logits)

        return train_loss, logits
    
    @tf.function
    def inference_step(self, x):
        # Only pass the samples through and collect the output vales
        return model(x, training=False)

    def train_one_epoch(self, step_counter, epoch):
        self.train_cindex_metric.reset_states()
        logits = tf.zeros([0, 1])
        for x, y in tqdm(self.train_ds):
            logits_batch = self.inference_step(x)
            logits = tf.concat([logits, logits_batch], 0)

        # Get the slides used in the training generator, so that the group_argtopk function can
        # rank the tiles on slide level (taking into account shuffeling)
        slides = self.train_ds.slide_tile_list[self.train_ds.indexes]
        slides = slides[:min(len(self.train_ds.slide_tile_list), logits.shape[0])]
        topk_idx = self.group_argtopk(slides, logits, k=10) # FOR TESTING PURPOSES, REMOVE!!!!

        self.train_ds.topk_dataset(topk_idx)

        for x, y in self.train_ds:
            train_loss, logits = self.train_one_step(
                x, y["label_event"], y["label_riskset"])

            step = int(step_counter)
            # if step == 0:
            #     func = self.train_one_step.get_concrete_function(
            #         x, y["label_event"], y["label_riskset"]
            #     )

            self.train_loss_metric.update_state(train_loss)
            self.train_cindex_metric.update_state(y, logits)
            step_counter.assign_add(1)


        # Display metrics
        mean_loss = self.train_loss_metric.result()
        train_cindex = self.train_cindex_metric.result()
        self.log_metrics([train_cindex['cindex']], 'train', 'pat', mean_loss)
        print(f"Epoch {step}: mean loss = {mean_loss:.4f}, train cindex = {train_cindex['cindex']:.4f}")
        # Reset training metrics
        self.train_loss_metric.reset_states()

        
  
    @tf.function
    def evaluate_one_step(self, x, y_event, y_riskset):
        y_event = tf.expand_dims(y_event, axis=1)
        val_logits = model(x, training=False)

        val_loss = self.loss_fn(y_true=[y_event, y_riskset], y_pred=val_logits)
        return val_loss, val_logits

    def evaluate(self, step_counter):
        self.val_cindex_metric.reset_states()

        logits = tf.zeros([0, 1])
        for x, y in self.val_ds:
            logits_batch = self.inference_step(x)
            logits = tf.concat([logits, logits_batch], 0)

        slides = self.val_ds.slide_tile_list[self.val_ds.indexes]
        slides = slides[:min(len(self.val_ds.slide_tile_list), logits.shape[0])]
        maxs = self.group_max(slides, logits, len(self.val_ds.slide_tile_list))  # check if len() is correct here + change this to mins

        self.val_ds.topk_dataset(maxs)
        # val_gen.topk_dataset(topk_idx)

        for x_val, y_val in self.val_ds:
            val_loss, val_logits = self.evaluate_one_step(
                x_val, y_val["label_event"], y_val["label_riskset"]
            )
            
            # Update val metrics
            self.val_loss_metric.update_state(val_loss)
            self.val_cindex_metric.update_state(y_val, val_logits)

        val_loss = self.val_loss_metric.result()

        # for step, (x_batch_val, y_batch_val) in enumerate(tqdm(val_gen)):
        #     loss_value = test_step(x_batch_val, y_batch_val)
        #     avg_loss_val += loss_value
        # avg_loss_val /= (step+1)
        self.val_loss_metric.reset_states()

        val_cindex = self.val_cindex_metric.result()
        self.log_metrics([val_cindex['cindex']], 'val', 'pat', val_loss)

        print(f"Validation: loss = {val_loss:.4f}, cindex = {val_cindex['cindex']:.4f}")

    def train_and_evaluate(self):
        ckpt = tf.train.Checkpoint(
            step=tf.Variable(0, dtype=tf.int64),
            optimizer=self.optimizer,
            model=self.model)
        
        for epoch in range(self.num_epochs):
            self.train_one_epoch(ckpt.step, epoch)
            self.evaluate(ckpt.step)
            self.train_ds.on_epoch_end()
            self.val_ds.on_epoch_end()

    def group_argtopk(self, groups, data, k=10):
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

        # minus to get reverse sort
        data = -data.numpy().ravel()
        order = np.lexsort((data, groups))
        groups = np.array(groups)
        groups = groups[order]
        data = data[order]
        index = np.empty(len(groups), 'bool')
        index[-k:] = True
        index[:-k] = groups[k:] != groups[:-k]
        return list(order[index])

    def group_max(self, groups, data, nmax):
        # minus to get reverse sort
        data = -data.numpy().ravel()
        out = np.empty(nmax)
        out[:] = np.nan
        order = np.lexsort((data, groups))
        groups = groups[order]
        data = data[order]
        index = np.empty(len(groups), 'bool')
        index[-1] = True
        index[:-1] = groups[1:] != groups[:-1]
        # out[groups[index]] = data[index]

        #  Adapted to return index!
        # return list(order[index])
        return np.where(index)[0]
    
    def log_metrics(self, metrics, split, prefix, loss):
        wandb.log({
            f"{prefix}_{split}_loss": loss,
            f"{prefix}_{split}_cindex": metrics[0],
            # f"{prefix}_{split}_f1": metrics[2],
            # f"{prefix}_{split}_balacc": metrics[1],
            # f"{prefix}_{split}_recall": metrics[4],
            # f"{prefix}_{split}_precision": metrics[3],
            # f"{prefix}_{split}_cnfmatrix": metrics[6],
            # f"{prefix}_{split}_auc": metrics[5]
            #f"{prefix}_{split}_fpr": metrics[7],
            #f"{prefix}_{split}_tpr": metrics[8]
        })


if __name__ == "__main__":
    # parsed_args = parse_arguments()
    patient_data = pd.read_excel(config.cohort_settings['cohort_file'], header=0, engine="openpyxl").set_index('ID').dropna()

    train_index, test_index, _, _ = train_test_split(
                                        patient_data.index,
                                        patient_data['Event'],
                                        test_size=0.2,
                                        random_state=0)
        
    # New model gets instantiated, so clear cache
    K.clear_session()
    print(f"  Train: index={train_index}")
    print(f"  Test:  index={test_index}")

    INPUT_PATH = './tiles-normalized/Training/Oversampled'

    train_gen = DatagenMILSurv(patient_data,
                           512,
                           INPUT_PATH,
                           batch_size=32,
                           train=True,
                           imagenet=False)
    val_gen = DatagenMILSurv(patient_data,
                         512,
                         INPUT_PATH,
                         batch_size=32,
                         train=False,
                         imagenet=False)

    model = create_classification_model(input_shape=(512, 512, 3))

    trainer = TrainAndEvaluateModel(
        model=model,
        model_dir=Path("ckpts-mnist-cnn"),
        train_dataset=train_gen,
        eval_dataset=val_gen,
        learning_rate=0.0001,
        num_epochs=5,
    )

    trainer.train_and_evaluate()

    print("all done")
