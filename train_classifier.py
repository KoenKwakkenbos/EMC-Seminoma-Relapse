import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import ResNet50
from datagenerator import Datagen
from sklearn.model_selection import train_test_split, StratifiedKFold
import tensorflow.keras.backend as K


def create_model(input_shape=(224, 224, 3)):
    inputs = layers.Input(input_shape, name ='input')
    resnet = ResNet50(include_top=False, weights=None, pooling='max')(inputs)
    x = layers.Dropout(0.5)(resnet)
    output = layers.Dense(1, activation='sigmoid')(x)
    model = keras.Model(inputs, output)
    model.layers[1].trainable = False
    print(model.summary())
    
    """
    model = keras.Sequential([
    layers.BatchNormalization(input_shape=input_shape),
    layers.Conv2D(64,kernel_size=(3,3)),
    keras.layers.ReLU(),
    layers.BatchNormalization(),
    layers.Conv2D(64,kernel_size=(3,3)),
    keras.layers.ReLU(),
    layers.BatchNormalization(),
    layers.MaxPooling2D(pool_size=((2,2))),
    layers.Conv2D(128,kernel_size=(3,3)),
    keras.layers.ReLU(),
    layers.BatchNormalization(),
    layers.Conv2D(128,kernel_size=(3,3)),
    keras.layers.ReLU(),
    layers.BatchNormalization(),
    layers.MaxPooling2D(pool_size=((2,2))),
    layers.Conv2D(256,kernel_size=(3,3)),
    keras.layers.ReLU(),
    layers.BatchNormalization(),
    layers.Conv2D(256,kernel_size=(3,3)),
    keras.layers.ReLU(),
    layers.BatchNormalization(),
    layers.MaxPooling2D(pool_size=((2,2))),
    layers.Conv2D(512,kernel_size=(3,3)),
    keras.layers.ReLU(),
    layers.BatchNormalization(),
    layers.Conv2D(512,kernel_size=(3,3)),
    keras.layers.ReLU(),
    layers.BatchNormalization(),
    layers.Conv2D(512,kernel_size=(3,3)),
    keras.layers.ReLU(),
    layers.BatchNormalization(),
    layers.MaxPooling2D(pool_size=((2,2))),
    layers.GlobalMaxPooling2D(),
    layers.Dense(512, activation='relu'),
    layers.Dense(256, activation='relu'),
    layers.Dense(1, activation='sigmoid'),
    ])
    """
    return model


def compute_class_weights(labels):
    labels = np.array(labels)

    negative_count = len(np.where(labels==0)[0])
    positive_count = len(np.where(labels==1)[0])
    total_count = negative_count + positive_count

    return {
        0: (1 / negative_count) * (total_count / 2),
        1: (1 / positive_count) * (total_count / 2),
    }


def train(train_gen, val_gen, weights):

    file_path = "./output_resnet_kfold/best_model_weights.h5"

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
        model = create_model((224, 224, 3))
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


if __name__ == "__main__":
    patient_data = pd.read_csv('./Seminoma_Outcomes_AnonSelection_20230124.csv', header=0).set_index('AnonPID')
    
    skf = StratifiedKFold(n_splits=3)
    skf.get_n_splits(patient_data.index, patient_data['Meta'])

    for i, (train_index, test_index) in enumerate(skf.split(patient_data.index, patient_data['Meta'])):
        K.clear_session()
        print(f"Fold {i}:")
        print(f"  Train: index={train_index}")
        print(f"  Test:  index={test_index}")

        print(patient_data.iloc[train_index])

        train_gen = Datagen(list(patient_data.iloc[train_index].index), patient_data['Meta'].iloc[train_index], 224, batch_size=32, train=True, imagenet=True)
        val_gen = Datagen(list(patient_data.iloc[test_index].index), patient_data['Meta'].iloc[test_index], 224, batch_size=32, train=False, imagenet=True)

        train_weights = compute_class_weights(patient_data['Meta'].iloc[train_index])

        model = train(train_gen, val_gen, train_weights)
