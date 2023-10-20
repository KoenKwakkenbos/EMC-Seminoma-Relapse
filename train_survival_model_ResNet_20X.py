import os
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from datagenerator_survival import Datagen
from sklearn.model_selection import train_test_split, StratifiedKFold
import tensorflow.keras.backend as K
import wandb

from models import *
import config_resnet_20 as config
from pathlib import Path
from survival_model import CoxPHLoss, CindexMetric
from tqdm import tqdm

print('starting')


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
        
        self.val_loss_list = []

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

        for epoch in range(1, self.num_epochs+1):
            train_loss, train_cindex = self.train_one_epoch(ckpt.step, epoch)

            # Run a validation loop at the end of each epoch.
            val_loss, val_cindex = self.evaluate(ckpt.step)
            
            # Log the results
            self.log_metrics([train_cindex['cindex']], 'train', 'img', train_loss)
            self.log_metrics([val_cindex['cindex']], 'val', 'img', val_loss)

            # Datagenerators end of epoch
            self.train_ds.on_epoch_end()
            self.val_ds.on_epoch_end()

            save_path = ckpt_manager.save()
            print(f"Saved checkpoint for step {ckpt.step.numpy()}: {save_path}")

            # Learning rate decay
            old_lr = self.optimizer.learning_rate
            new_lr = old_lr * 0.95**epoch
            K.set_value(self.optimizer.learning_rate, new_lr)

            # Early stopping
            if len(self.val_loss_list) > 15:
                if self.val_loss_list[-1] > self.val_loss_list[-15]:
                    print("Loss stopped decreasing, early stopping activated")
                    break

    def train_one_epoch(self, step_counter, epoch):
        self.train_cindex_metric.reset_states()
        for x, y in tqdm(self.train_ds):
            train_loss, logits = self.train_one_step(
                x, y["label_event"], y["label_riskset"])

            step = int(step_counter)

            # Update training metrics.
            self.train_loss_metric.update_state(train_loss)
            self.train_cindex_metric.update_state(y, logits)
            step_counter.assign_add(1)
 
        mean_loss = self.train_loss_metric.result()
        train_cindex = self.train_cindex_metric.result()

        # Print epoch loss
        print(f"Epoch {epoch}: mean loss = {mean_loss:.4f}, train c-index: {train_cindex['cindex']:.4f}")
        # Reset training metrics
        self.train_loss_metric.reset_states()

        return mean_loss, train_cindex
            
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
        # Add val_loss to monitor: 
        self.val_loss_list.append(val_loss)
        
        self.val_loss_metric.reset_states()
        
        val_cindex = self.val_cindex_metric.result()

        print(f"Validation: loss = {val_loss:.4f}, cindex = {val_cindex['cindex']:.4f}")

        return val_loss, val_cindex

    def log_metrics(self, metrics, split, prefix, loss):
        wandb.log({
            f"{prefix}_{split}_loss": loss,
            f"{prefix}_{split}_cindex": metrics[0],
        })


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
        project="ResNet50_Fully_Supervised_Survival",

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

    train_gen = Datagen(patient_data_train,
                        config.model_settings['tile_size'], 
                        config.cohort_settings['data_path'], 
                        batch_size=config.train_settings['batch_size'], 
                        train=True, 
                        imagenet=True)

    val_gen = Datagen(patient_data_val,
                    config.model_settings['tile_size'], 
                    config.cohort_settings['data_path'],
                    batch_size=config.train_settings['batch_size'], 
                    train=False, 
                    imagenet=True)

    model = create_imagenet_model(input_shape=(config.model_settings['tile_size'], config.model_settings['tile_size'], 3), trainable=False)
    print(model.summary())

    trainer = TrainAndEvaluateModel(
        model=model,
        model_dir=Path(f"ResNet50_FSS_20_fold_{i+1}"),
        train_dataset=train_gen,
        eval_dataset=val_gen,
        learning_rate=config.train_settings['learning_rate'],
        num_epochs=config.train_settings['epochs'],
    )

    trainer.train_and_evaluate()

    print('done')
    wandb.join()
