import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from datagenerator import MILdatagen


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

    encoder = layers.BatchNormalization()(x)

    # Decoder
    x = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = layers.UpSampling2D((2, 2))(encoder)
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
    file_path = "./tmp/best_model_weights.h5"

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
        callbacks=[early_stopping, model_checkpoint],
        verbose=1,
    )

    # Load best weights.
    model.load_weights(file_path)
    
    return model


if __name__ == "__main__":
    model = create_model((224, 224, 3))
    
    train_gen = MILdatagen(['./output6/TZ_08_G_HE_1'], 224, batch_size =32, train=False)
    val_gen = MILdatagen(['./output6/TZ_10_D_HE_1'], 224, batch_size =32, train=False)

    train(model, train_gen, val_gen)