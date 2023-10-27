import os
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import StratifiedKFold
import tensorflow.keras.backend as K
from tensorflow.keras.applications.densenet import preprocess_input

from survival_models import *
from pathlib import Path
from tqdm import tqdm
import numpy as np

# imports
from tensorflow.keras import layers
from tensorflow.keras.applications import DenseNet121
import random

def create_imagenet_model(input_shape=(224, 224, 3), num_clinical_features=3, trainable=False):
    """Creates a ResNet50-based binary classification model.

    Parameters:
    - input_shape (tuple, optional): The input shape of the model. Defaults to (224, 224, 3).
    - trainable (bool, optional): Whether the ResNet50 layers should be trainable. Defaults to False.

    Returns:
    - keras.Model: The ResNet50-based binary classification model.
    """
    image_input = layers.Input(input_shape, name='input')
    clinical_input = layers.Input(shape=(num_clinical_features,), name='clinical_input')

#     x = layers.BatchNormalization()(image_input)
    
    rotation = layers.RandomRotation(factor=(-0.5, 0.5))(image_input)

    resnet = DenseNet121(input_shape=input_shape,
                      include_top=False,
                      weights='imagenet')
    resnet.trainable = trainable

    global_average_layer = layers.GlobalAveragePooling2D()

    x = resnet(rotation, training=trainable)
    x = global_average_layer(x)

    # clin_in = layers.BatchNormalization()(clinical_input)
    concatenated_features = layers.concatenate([x, clinical_input])
#     y = layers.BatchNormalization()(concatenated_features)
    y = layers.Dropout(0.3)(concatenated_features)
#     y = layers.Dense(512, activation='relu')(y)
    # y = layers.Dropout(0.4)(y)
    # y = layers.Dense(32, activation='relu')(y)
    output = layers.Dense(1, activation='linear')(y)
    model = keras.Model(inputs=[image_input, clinical_input], outputs=output)

    # model.layers[1].trainable = trainable

    return model


def train_val_split(tile_list, df_train, df_val, clinical_vars):
    tile_list_train = [file for file in tile_list
                       if extract_identifier(os.path.basename(file)) in df_train.index]
    tile_list_val = [file for file in tile_list
                     if extract_identifier(os.path.basename(file)) in df_val.index]

    train_ids = [extract_identifier(os.path.basename(file)) for file in tile_list_train]
    clin_train = np.array(df_train.loc[train_ids][clinical_vars], dtype=np.float32)
    event_train = np.array(df_train.loc[train_ids]['Event'], dtype=np.int32)
    time_train = np.array(df_train.loc[train_ids]['Time'], dtype=np.float32)

    val_ids = [extract_identifier(os.path.basename(file)) for file in tile_list_val]
    clin_val = np.array(df_val.loc[val_ids][clinical_vars], dtype=np.float32)
    event_val = np.array(df_val.loc[val_ids]['Event'], dtype=np.int32)
    time_val = np.array(df_val.loc[val_ids]['Time'], dtype=np.float32)

    return tile_list_train, (clin_train, event_train, time_train), tile_list_val, (clin_val, event_val, time_val)


def extract_identifier(filename):
    # Attempt extraction using both hyphen and underscore separators
    separators = ['-', '_']
    for separator in separators:
        parts = filename.split(separator)
        if len(parts) >= 3:
            identifier = separator.join(parts[:3])
            return identifier
    return None

def _make_riskset(images, clinical_vars, time: np.ndarray, labels_event) -> np.ndarray:
    """Compute mask that represents each sample's risk set.

    Parameters
    ----------
    time : np.ndarray, shape=(n_samples,)
        Observed event time sorted in descending order.

    Returns
    -------
    risk_set : np.ndarray, shape=(n_samples, n_samples)
        Boolean matrix where the `i`-th row denotes the
        risk set of the `i`-th instance, i.e. the indices `j`
        for which the observer time `y_j >= y_i`.
    """
    assert time.ndim == 1, "expected 1D array"

    # sort in descending order
    o = np.argsort(-time, kind="mergesort")
    n_samples = len(time)
    risk_set = np.zeros((n_samples, n_samples), dtype=np.bool_)
    for i_org, i_sort in enumerate(o):
        ti = time[i_sort]
        k = i_org
        while k < n_samples and ti == time[o[k]]:
            k += 1
        risk_set[i_sort, o[:k]] = True
    return images, clinical_vars, time, labels_event, risk_set


def _process_batch(images, clinical_vars, labels_time, labels_event, risk_set):
    labels = {
            "label_event": labels_event,
            "label_time": labels_time,
            "label_riskset": risk_set
    }
    return (images, clinical_vars), labels


def prepare_input(tile_path, clinical_vars):
    raw = tf.io.read_file(tile_path)
    image = tf.io.decode_image(raw)
    image = tf.cast(image, tf.float32)

    image = preprocess_input(image)

    return (image, clinical_vars), tile_path

# STRATIFIED 5-FOLD CROSS VALIDATION
patient_data = pd.read_excel('/data/scratch/kkwakkenbos/train_val_cohort.xlsx', header=0, engine="openpyxl").set_index('ID').dropna()

patient_data = patient_data.drop(patient_data[patient_data['Synchronous'] == 1].index)

skf = StratifiedKFold(n_splits=5)
skf.get_n_splits(patient_data.index, patient_data['Event'])


for i, (train_index, val_index) in enumerate(skf.split(patient_data.index, patient_data['Event'])):
    K.clear_session()
    print(f"Fold {i+1}:")
    print(f"  Train: index={train_index}")
    print(f"  Test:  index={val_index}")

    patient_data_train = patient_data.iloc[train_index].copy()
    patient_data_val = patient_data.iloc[val_index].copy()

    clinical_vars = ['RTI', 'LVI', 'Size']

    tile_list = [os.path.join(subdir, file) 
                 for subdir, dirs, files in os.walk('/data/scratch/kkwakkenbos/Tiles_512_10x_normalized')
                 for file in files if file.lower().endswith(('.png', '.jpg'))]


    _, _, tiles_val, labels_val = train_val_split(tile_list, patient_data_train, patient_data_val, ['LVI', 'RTI', 'Size', 'Treatment'])

    model = create_imagenet_model(input_shape=(512, 512, 3), num_clinical_features=4, trainable=False)
    print(model.summary())

    checkpoint = tf.train.Checkpoint(model=model)
    checkpoint_manager = tf.train.CheckpointManager(checkpoint, f'/trinity/home/kkwakkenbos/repositories/October2023Runs/DenseNet/DenseNet121_FSS_10newdl_fold_{i+1}', max_to_keep=5)

    # Restore the latest checkpoint
    latest_checkpoint = checkpoint_manager.latest_checkpoint
    if latest_checkpoint:
        checkpoint.restore(latest_checkpoint)
        print(f"Model restored from {latest_checkpoint}")
    else:
        print("No checkpoint found. You may need to train the model first.")
        continue

    inference_preds = []
    agg_tiles = []

    dataset = tf.data.Dataset.from_tensor_slices((tiles_val, labels_val[0]))
    dataset = dataset.map(prepare_input, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(128)
    dataset = dataset.apply(tf.data.experimental.ignore_errors())

    for input, tile in tqdm(dataset):
        inference_preds.extend(model(input, training=False).numpy())
        agg_tiles.extend(tile.numpy())


    pd.DataFrame({'File': np.array(agg_tiles).ravel(), 'Prediction': np.array(inference_preds).ravel()}).to_csv(f'inference_dict_fold_{i+1}.csv')
    print('done')


# inference_preds = []
# agg_tiles = []

# dataset = tf.data.Dataset.from_tensor_slices((tiles_val[:64], labels_val[0][:64]))
# dataset = dataset.map(prepare_input, num_parallel_calls=tf.data.experimental.AUTOTUNE)
# dataset = dataset.batch(32)
# dataset = dataset.apply(tf.data.experimental.ignore_errors())

# for input, tile in tqdm(dataset):
#     inference_preds.extend(model(input, training=False).numpy())
#     agg_tiles.extend(tile.numpy())

# pd.DataFrame({'File': np.array(agg_tiles).ravel(), 'Prediction': np.array(inference_preds).ravel()}).to_csv(f'inference_dict_fold_{i+1}.csv')
# print('done')
