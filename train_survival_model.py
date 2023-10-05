import os
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from datagenerator_survival import Datagen
from sklearn.model_selection import train_test_split, StratifiedKFold
import tensorflow.keras.backend as K

from models import create_classification_model
from helpers import tar_path, dir_path
import config
from pathlib import Path
from survival_model import CoxPHLoss, CindexMetric

# CHECK DISTRIBUTED TRAINING!!


import tensorflow.compat.v2.summary as summary
from tensorflow.python.ops import summary_ops_v2


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
        # ckpt_manager = tf.train.CheckpointManager(
            # ckpt, str(self.model_dir), max_to_keep=2)

        # if ckpt_manager.latest_checkpoint:
            # ckpt.restore(ckpt_manager.latest_checkpoint)
            # print(f"Latest checkpoint restored from {ckpt_manager.latest_checkpoint}.")

        # train_summary_writer = summary.create_file_writer(
        #     str(self.model_dir / "train"))
        # val_summary_writer = summary.create_file_writer(
        #     str(self.model_dir / "valid"))

        for epoch in range(self.num_epochs):
            # with train_summary_writer.as_default():
            self.train_one_epoch(ckpt.step)

            # Run a validation loop at the end of each epoch.
            # with val_summary_writer.as_default():
            self.evaluate(ckpt.step)

            # Datagenerators end of epoch
            self.train_ds.on_epoch_end()
            self.val_ds.on_epoch_end()

        # save_path = ckpt_manager.save()
        # print(f"Saved checkpoint for step {ckpt.step.numpy()}: {save_path}")

    def train_one_epoch(self, step_counter):
        for x, y in self.train_ds:
            train_loss, logits = self.train_one_step(
                x, y["label_event"], y["label_riskset"])

            step = int(step_counter)
            if step == 0:
                # see https://stackoverflow.com/questions/58843269/display-graph-using-tensorflow-v2-0-in-tensorboard
                func = self.train_one_step.get_concrete_function(
                    x, y["label_event"], y["label_riskset"])
                # summary_ops_v2.graph(func.graph, step=0)

            # Update training metric.
            self.train_loss_metric.update_state(train_loss)

            # Log every 200 batches.
            if step % 1 == 0:
                # Display metrics
                mean_loss = self.train_loss_metric.result()
                print(f"step {step}: mean loss = {mean_loss:.4f}")
                # save summaries
                summary.scalar("loss", mean_loss, step=step_counter)
                # Reset training metrics
                self.train_loss_metric.reset_states()

            step_counter.assign_add(1)

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
        # summary.scalar("loss",
        #                val_loss,
        #                step=step_counter)
        self.val_loss_metric.reset_states()
        
        val_cindex = self.val_cindex_metric.result()
        # for key, value in val_cindex.items():
        #     summary.scalar(key, value, step=step_counter)

        print(f"Validation: loss = {val_loss:.4f}, cindex = {val_cindex['cindex']:.4f}")


# patient_data = pd.read_csv('./Seminoma_Outcomes_Anon.csv', header=0).set_index('AnonPID')
patient_data = pd.read_excel(config.cohort_settings['cohort_file'], header=0, engine="openpyxl").set_index('ID').dropna()

if not config.cohort_settings['synchronous']:
    patient_data = patient_data.drop(patient_data[patient_data['Synchronous'] == 1].index)
if not config.cohort_settings['treatment']:
    patient_data = patient_data.drop(patient_data[patient_data['Treatment'] == 1].index)

print(patient_data.head(10))

train_index, test_index, _, _ = train_test_split(
    patient_data.index, patient_data['Event'], test_size=0.2, random_state=0)


print(f"  Train: index={train_index}")
print(f"  Test:  index={test_index}")

print(patient_data.loc[train_index])

INPUT_PATH = './tiles_TCGA_normalized'

train_gen = Datagen(list(patient_data.loc[train_index].index), 
                    patient_data['Event'].loc[train_index],
                    patient_data['Time'].loc[train_index], 
                    224, 
                    INPUT_PATH, 
                    batch_size=32, 
                    train=True, 
                    imagenet=True)
val_gen = Datagen(list(patient_data.loc[test_index].index), 
                    patient_data['Event'].loc[test_index], 
                    patient_data['Time'].loc[test_index], 
                    224, 
                    INPUT_PATH, 
                    batch_size=32, 
                    train=False, 
                    imagenet=True)

model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(6, kernel_size=(5, 5), activation='relu', name='conv_1'),
        tf.keras.layers.Conv2D(6, kernel_size=(5, 5), activation='relu', name='conv_2'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(16, (5, 5), activation='relu', name='conv_3'),
        tf.keras.layers.Conv2D(16, (5, 5), activation='relu', name='conv_4'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(32, (5, 5), activation='relu', name='conv_5'),
        tf.keras.layers.Conv2D(32, (5, 5), activation='relu', name='conv_6'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(120, activation='relu', name='dense_1'),
        tf.keras.layers.Dense(84, activation='relu', name='dense_2'),
        tf.keras.layers.Dense(1, activation='linear', name='dense_3')
])

trainer = TrainAndEvaluateModel(
    model=model,
    model_dir=Path("ckpts-mnist-cnn"),
    train_dataset=train_gen,
    eval_dataset=val_gen,
    learning_rate=0.0001,
    num_epochs=15,
)

trainer.train_and_evaluate()

print('done')