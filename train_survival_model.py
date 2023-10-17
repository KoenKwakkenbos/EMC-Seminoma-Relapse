import os
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from datagenerator_survival import Datagen
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.utils import resample
import tensorflow.keras.backend as K
import wandb

from models import create_classification_model
from helpers import tar_path, dir_path
import config
from pathlib import Path
from survival_model import CoxPHLoss, CindexMetric
from tqdm import tqdm
import random

# CHECK DISTRIBUTED TRAINING!!


import tensorflow.compat.v2.summary as summary
from tensorflow.python.ops import summary_ops_v2


print('starting')
# Start a run, tracking hyperparameters
wandb.init(
    # set the wandb project where this run will be logged
    project="fully-supervised-survival",

    # track hyperparameters and run metadata with wandb.config
    config={
        "layer_1": 512,
        "activation_1": "relu",
        "dropout": random.uniform(0.01, 0.80),
        "layer_2": 10,
        "activation_2": "softmax",
        "optimizer": "sgd",
        "loss": "sparse_categorical_crossentropy",
        "metric": "accuracy",
        "epoch": 8,
        "batch_size": 256
    }
)

class TrainAndEvaluateModel:
    def __init__(self, model, model_dir, train_dataset, eval_dataset,
                 learning_rate, num_epochs):
        self.num_epochs = num_epochs
        self.model_dir = model_dir

        self.model = model

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
            logits = self.model(x, training=True)

            train_loss = self.loss_fn(y_true=[y_event, y_riskset], y_pred=logits)

        with tf.name_scope("gradients"):
            grads = tape.gradient(train_loss, self.model.trainable_weights)
            self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
        return train_loss, logits

    def train_and_evaluate(self):
        ckpt = tf.train.Checkpoint(
            step=tf.Variable(0, dtype=tf.int64),
            optimizer=self.optimizer,
            model=self.model)
        ckpt_manager = tf.train.CheckpointManager(
            ckpt, str(self.model_dir), max_to_keep=2)

        if ckpt_manager.latest_checkpoint:
            ckpt.restore(ckpt_manager.latest_checkpoint)
            print(f"Latest checkpoint restored from {ckpt_manager.latest_checkpoint}.")

        for epoch in range(self.num_epochs):
            self.train_one_epoch(ckpt.step, epoch)

            # Run a validation loop at the end of each epoch.
            self.evaluate(ckpt.step)

            # Datagenerators end of epoch
            self.train_ds.on_epoch_end()
            self.val_ds.on_epoch_end()

        save_path = ckpt_manager.save()
        print(f"Saved checkpoint for step {ckpt.step.numpy()}: {save_path}")

    def train_one_epoch(self, step_counter, epoch):
        self.train_cindex_metric.reset_states()
        for x, y in tqdm(self.train_ds):
            train_loss, logits = self.train_one_step(
                x, y["label_event"], y["label_riskset"])
            

            step = int(step_counter)
            # if step == 0:
                # see https://stackoverflow.com/questions/58843269/display-graph-using-tensorflow-v2-0-in-tensorboard
                # func = self.train_one_step.get_concrete_function(
                #     x, y["label_event"], y["label_riskset"])

            # Update training metrics.
            self.train_loss_metric.update_state(train_loss)
            self.train_cindex_metric.update_state(y, logits)
            step_counter.assign_add(1)
 
        mean_loss = self.train_loss_metric.result()
        train_cindex = self.train_cindex_metric.result()
        self.log_metrics([train_cindex['cindex']], 'train', 'img', mean_loss)
        # Print epoch loss
        print(f"Epoch {epoch}: mean loss = {mean_loss:.4f}")
        # Reset training metrics
        self.train_loss_metric.reset_states()

            
    @tf.function
    def evaluate_one_step(self, x, y_event, y_riskset):
        y_event = tf.expand_dims(y_event, axis=1)
        val_logits = self.model(x, training=False)
        val_loss = self.loss_fn(y_true=[y_event, y_riskset], y_pred=val_logits)
        return val_loss, val_logits

    def evaluate(self, step_counter):
        self.val_cindex_metric.reset_states()
        
        for x_val, y_val in self.val_ds:
            val_loss, val_logits = self.evaluate_one_step(
                x_val, y_val["label_event"], y_val["label_riskset"])

            # Update val metrics
            self.val_loss_metric.update_state(val_loss)
            self.val_cindex_metric.update_state(y_val, val_logits)

        val_loss = self.val_loss_metric.result()

        self.val_loss_metric.reset_states()
        
        val_cindex = self.val_cindex_metric.result()
        self.log_metrics([val_cindex['cindex']], 'val', 'img', val_loss)

        print(f"Validation: loss = {val_loss:.4f}, cindex = {val_cindex['cindex']:.4f}")

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


patient_data = pd.read_excel(config.cohort_settings['cohort_file'], header=0, engine="openpyxl").set_index('ID').dropna()

if not config.cohort_settings['synchronous']:
    patient_data = patient_data.drop(patient_data[patient_data['Synchronous'] == 1].index)
if not config.cohort_settings['treatment']:
    patient_data = patient_data.drop(patient_data[patient_data['Treatment'] == 1].index)

print(patient_data.head(10))

train_index, val_index, _, _ = train_test_split(
    patient_data.index, patient_data['Event'], stratify=patient_data['Event'], test_size=0.2, random_state=0)


patient_data_train = patient_data.loc[train_index].copy()
patient_data_val = patient_data.loc[val_index].copy()

print(f"  Train: index={train_index}")
print(f"  Test:  index={val_index}")

clinical_vars = ['RTI', 'LVI', 'Size']

train_gen = Datagen(patient_data_train,
                    512, 
                    config.cohort_settings['data_path'], 
                    batch_size=32, 
                    train=True, 
                    imagenet=True)

val_gen = Datagen(patient_data_val,
                  512, 
                  config.cohort_settings['data_path'],
                  batch_size=32, 
                  train=False, 
                  imagenet=True)

model = create_classification_model(input_shape=(512, 512, 3), trainable=False)


trainer = TrainAndEvaluateModel(
    model=model,
    model_dir=Path("surv_model_first_try"),
    train_dataset=train_gen,
    eval_dataset=val_gen,
    learning_rate=0.001,
    num_epochs=15,
)

trainer.train_and_evaluate()

print('done')