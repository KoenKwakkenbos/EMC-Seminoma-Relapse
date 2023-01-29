import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from datagenerator import MILdatagen
from sklearn.model_selection import train_test_split


def create_model(input_shape):
    input_img = keras.Input(shape=input_shape)

    # Encoder
    x = layers.BatchNormalization()(input_img)
    x = layers.Conv2D(16, (3, 3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(16, (3, 3), padding='same', strides = (2,2), activation='relu')(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(32, (3, 3), padding='same', strides = (2,2), activation='relu')(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(64, (3, 3), padding='same', strides = (2,2), activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(64, (3, 3), padding='same', strides = (2,2), activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(128, (3, 3), padding='same', strides = (2,2), activation='relu')(x)
    x = layers.BatchNormalization(name='encoder_final_layer')(x)

    encoder = x

    # Decoder
    x = layers.Conv2D(128, (3, 3), padding='same', activation='relu')(encoder)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.UpSampling2D((2, 2))(x)

    x = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.UpSampling2D((2, 2))(x)

    x = layers.Conv2D(16, (3, 3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(16, (3, 3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.UpSampling2D((2, 2))(x)

    decoder = layers.Conv2D(3, (3, 3), padding='same', activation='sigmoid')(x)

    model = keras.Model(input_img, decoder)

    return model


def train(model, train_gen, val_gen):
    file_path = "./output/best_model_weights.h5"

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
    
    def SSIMLoss(y_true, y_pred):
        return 0.5*(1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, 1.0))) + 0.5*keras.losses.mean_absolute_error(y_true, y_pred) + 0.5*keras.losses.mean_squared_error(y_true, y_pred)

    # Compile model.
    model.compile(
        optimizer="adam", loss=[SSIMLoss]
    )

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


if __name__ == "__main__":
    model = create_model((224, 224, 3))

    patient_data = pd.read_csv('./Seminoma_Outcomes_AnonSelection_20230124.csv', header=0)
    print(patient_data.head())

    
    pat_train, pat_val, y_train, y_val = train_test_split(
        patient_data['AnonPID'], patient_data['Meta'], test_size=0.25, random_state=42,
        stratify=patient_data['Meta']
    )

    print(list(pat_train))


    train_gen = MILdatagen(list(pat_train), 224, batch_size=32, train=True)
    val_gen = MILdatagen(list(pat_val), 224, batch_size=32, train=False)

    model = train(model, train_gen, val_gen)

    encoder = keras.Model(model.input, model.layers[17].output)


    train_gen = MILdatagen(list(pat_train), 224, batch_size=32, train=False)
    val_gen = MILdatagen(list(pat_val), 224, batch_size=32, train=False)

    X_train_embedded = encoder.predict(train_gen)
    X_val_embedded = encoder.predict(val_gen)

    np.save('./output/X_train_embedded.npy', X_train_embedded)
    np.save('./output/X_val_embedded.npy', X_val_embedded)
