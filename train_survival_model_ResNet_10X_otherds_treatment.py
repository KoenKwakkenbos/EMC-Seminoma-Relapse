import os
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tf_dataset import get_datagenerator
from sklearn.model_selection import train_test_split, StratifiedKFold
import tensorflow.keras.backend as K
import wandb

from models import *
import config_resnet_10 as config
from pathlib import Path
from survival_model import CoxPHLoss, CindexMetric
from tqdm import tqdm
import numpy as np

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


def train_val_split(tile_list, df_train, df_val):
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
    clin_train = np.array(df_train.loc[train_ids][['LVI', 'RTI', 'Size', 'Treatment']], dtype=np.float32)
    event_train = np.array(df_train.loc[train_ids]['Event'], dtype=np.int32)
    time_train = np.array(df_train.loc[train_ids]['Time'], dtype=np.float32)

    val_ids = [extract_identifier(os.path.basename(file)) for file in tile_list_val]
    clin_val = np.array(df_val.loc[val_ids][['LVI', 'RTI', 'Size', 'Treatment']], dtype=np.float32)
    event_val = np.array(df_val.loc[val_ids]['Event'], dtype=np.int32)
    time_val = np.array(df_val.loc[val_ids]['Time'], dtype=np.float32)

    return tile_list_train, (clin_train, event_train, time_train), tile_list_val, (clin_val, event_val, time_val)


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
            # self.train_ds = self.train_ds.shuffle()
            # self.val_ds = self.val_ds.shuffle(buffer_size=1024)
            save_path = ckpt_manager.save()
            print(f"Saved checkpoint for step {ckpt.step.numpy()}: {save_path}")

            # Learning rate decay
            # new_lr = config.train_settings['learning_rate'] * np.exp(-0.1*epoch)
            # K.set_value(self.optimizer.learning_rate, new_lr)
            # print(f"Learning rate set to: {self.optimizer.learning_rate}")
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
        
        for x_val, y_val in tqdm(self.val_ds):
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
        project="ResNet50_Fully_Supervised_Survival_newdatagen",

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

    tiles_train, labels_train, tiles_val, labels_val = train_val_split(tile_list, patient_data_train, patient_data_val)

    print(len(tiles_train))
    print(len(tiles_val))

    train_gen = get_datagenerator(tiles_train, labels_train[0], labels_train[1], labels_train[2], config.train_settings['batch_size'], train=True, imagenet=True)
    val_gen = get_datagenerator(tiles_val, labels_val[0], labels_val[1], labels_val[2], config.train_settings['batch_size'], train=False, imagenet=True)

    model = create_imagenet_model(input_shape=(config.model_settings['tile_size'], config.model_settings['tile_size'], 3), num_clinical_features=4, trainable=False)
    print(model.summary())

    trainer = TrainAndEvaluateModel(
        model=model,
        model_dir=Path(f"ResNet50_FSS_10newdl_treatment_fold_{i+1}"),
        train_dataset=train_gen,
        eval_dataset=val_gen,
        learning_rate=config.train_settings['learning_rate'],
        num_epochs=config.train_settings['epochs'],
    )

    trainer.train_and_evaluate()

    print('done')
    # wandb.join()
