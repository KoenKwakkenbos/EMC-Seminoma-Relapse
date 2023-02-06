import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from datagenerator import MILdatagen
from sklearn.model_selection import train_test_split


"""
def create_model(input_shape):
    input_img = keras.Input(shape=input_shape)

    # Encoder
    x = layers.Conv2D(16, (3, 3), padding='same', activation='relu')(input_img)
    x = layers.Conv2D(16, (3, 3), padding='same', strides = (2,2), activation='relu')(x)

    x = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(x)
    x = layers.Conv2D(32, (3, 3), padding='same', strides = (2,2), activation='relu')(x)

    x = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = layers.Conv2D(64, (3, 3), padding='same', strides = (2,2), activation='relu')(x)
    x = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = layers.Conv2D(64, (3, 3), padding='same', strides = (2,2), activation='relu', name='encoder_final_layer')(x)

    encoder = x

    # Decoder
    x = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = layers.UpSampling2D((2, 2))(x)

    x = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(x)
    x = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(x)
    x = layers.UpSampling2D((2, 2))(x)

    x = layers.Conv2D(16, (3, 3), padding='same', activation='relu')(x)
    x = layers.Conv2D(16, (3, 3), padding='same', activation='relu')(x)
    x = layers.UpSampling2D((2, 2))(x)

    decoder = layers.Conv2D(3, (3, 3), padding='same', activation='sigmoid')(x)

    model = keras.Model(input_img, decoder)
    return model

def create_model(input_shape):
    input_img = layers.Input(shape=input_shape)  # adapt this if using `channels_first` image data format

    encoder = layers.Conv2D(filters=16, kernel_size=(3,3),strides=1, padding='same')(input_img)#x^2*16
    encoder = layers.LeakyReLU()(encoder)
    encoder = layers.Conv2D(filters=32, kernel_size=(3,3),strides=1, padding='same')(encoder)#x^2*32
    encoder = layers.LeakyReLU()(encoder)
    encoder = layers.BatchNormalization()(encoder)
    encoder = layers.Conv2D(filters=64, kernel_size=(3,3),strides=2, padding='same')(encoder)#(x/2)^2*64
    encoder = layers.LeakyReLU()(encoder)
    encoder = layers.BatchNormalization()(encoder)
    encoder = layers.MaxPooling2D()(encoder)#(x/4)^2*64
    encoder = layers.Conv2D(filters=128, kernel_size=(3,3),strides=2, padding='same')(encoder)#(x/8)^2*128
    encoder = layers.LeakyReLU()(encoder)
    encoder = layers.BatchNormalization()(encoder)
    encoder = layers.MaxPooling2D()(encoder)#(x/16)^2*64
    encoder = layers.Conv2D(filters=64, kernel_size=(3,3),strides=1, padding='same')(encoder)#(x/16)^2*64
    encoder = layers.LeakyReLU()(encoder)
    encoder = layers.BatchNormalization()(encoder)
    encoder = layers.Conv2D(filters=32, kernel_size=(3,3),strides=1, padding='same')(encoder)#(x/16)^2*32
    encoder = layers.LeakyReLU()(encoder)


    bottleneck = layers.Conv2D(filters=16, kernel_size=(1,1),strides=1, padding='same')(encoder)#(x/16)^2*16
    bottleneck = layers.LeakyReLU()(bottleneck)


    decoder = layers.Conv2D(filters=32, kernel_size=(1,1),strides=1, padding='same')(bottleneck)#(x/16)^2*32
    decoder = layers.LeakyReLU()(decoder)
    decoder = layers.Conv2D(filters=64, kernel_size=(3,3),strides=1, padding='same')(decoder)#(x/16)^2*64
    decoder = layers.LeakyReLU()(decoder)
    decoder = layers.BatchNormalization()(decoder)
    decoder = layers.UpSampling2D()(decoder)#(x/8)^2*64
    decoder = layers.Conv2DTranspose(filters=128, kernel_size=(3,3),strides=2, padding='same')(decoder)#(x/4)^2*128
    decoder = layers.LeakyReLU()(decoder)
    decoder = layers.BatchNormalization()(decoder)
    decoder = layers.UpSampling2D()(decoder)#(x/2)^2*128
    decoder = layers.Conv2DTranspose(filters=64, kernel_size=(3,3),strides=2, padding='same')(decoder)#x^2*64
    decoder = layers.LeakyReLU()(decoder)
    decoder = layers.BatchNormalization()(decoder)
    decoder = layers.Conv2D(filters=32, kernel_size=(3,3),strides=1, padding='same')(decoder)#x^2*32
    decoder = layers.LeakyReLU()(decoder)
    decoder = layers.BatchNormalization()(decoder)
    decoder = layers.Conv2D(filters=16, kernel_size=(3,3),strides=1, padding='same')(decoder)#x^2*16
    decoder = layers.LeakyReLU()(decoder)
    decoder = layers.Conv2D(filters=3, kernel_size=(3,3),strides=1, padding='same')(decoder)#x^2*3
    decoded = layers.LeakyReLU()(decoder)

    model = keras.Model(input_img, decoded)
    return model
"""

def create_model(input_shape = (200,200,1)):
    input_size = input_shape
    #################################
    # Encoder
    #################################
    inputs = layers.Input(input_size, name = 'input')

    conv1 = layers.Conv2D(64, (3, 3),  padding = 'same', name ='conv1_1')(inputs)
    conv1 = layers.LeakyReLU()(conv1)
    conv1 = layers.BatchNormalization()(conv1)
    conv1 = layers.Conv2D(64, (3, 3),  padding = 'same', name ='conv1_2')(conv1)
    conv1 = layers.LeakyReLU()(conv1)
    conv1 = layers.BatchNormalization()(conv1)
    pool1 = layers.MaxPooling2D(pool_size = (2,2), strides = (2,2), name = 'pool_1')(conv1)

    conv2 = layers.Conv2D(128, (3, 3),  padding = 'same', name ='conv2_1')(pool1)
    conv2 = layers.LeakyReLU()(conv2)
    conv2 = layers.BatchNormalization()(conv2)
    conv2 = layers.Conv2D(128, (3, 3),  padding = 'same', name ='conv2_2')(conv2)
    conv2 = layers.LeakyReLU()(conv2)
    conv2 = layers.BatchNormalization()(conv2)
    pool2 = layers.MaxPooling2D(pool_size = (2,2), strides = (2,2), name = 'pool_2')(conv2)
    
    conv3 = layers.Conv2D(256, (3, 3),  padding = 'same', name ='conv3_1')(pool2)
    conv3 = layers.LeakyReLU()(conv3)
    conv3 = layers.BatchNormalization()(conv3)
    conv3 = layers.Conv2D(256, (3, 3),  padding = 'same', name ='conv3_2')(conv3)
    conv3 = layers.LeakyReLU()(conv3)
    conv3 = layers.BatchNormalization()(conv3)
    conv3 = layers.Conv2D(256, (3, 3),  padding = 'same', name ='conv3_3')(conv3)
    conv3 = layers.LeakyReLU()(conv3)
    conv3 = layers.BatchNormalization()(conv3)
    pool3 = layers.MaxPooling2D(pool_size = (2,2), strides = (2,2), name = 'pool_3')(conv3)
    
    conv4 = layers.Conv2D(512, (3, 3),  padding = 'same', name ='conv4_1')(pool3)
    conv4 = layers.LeakyReLU()(conv4)
    conv4 = layers.BatchNormalization()(conv4)
    conv4 = layers.Conv2D(512, (3, 3),  padding = 'same', name ='conv4_2')(conv4)
    conv4 = layers.LeakyReLU()(conv4)
    conv4 = layers.BatchNormalization()(conv4)
    conv4 = layers.Conv2D(512, (3, 3),  padding = 'same', name ='conv4_3')(conv4)
    conv4 = layers.LeakyReLU()(conv4)
    conv4 = layers.BatchNormalization()(conv4)
    pool4 = layers.MaxPooling2D(pool_size = (2,2), strides = (2,2), name = 'pool_4')(conv4)

    conv5 = layers.Conv2D(512, (3, 3),  padding = 'same', name ='conv5_1')(pool4)
    conv5 = layers.LeakyReLU()(conv5)
    conv5 = layers.BatchNormalization()(conv5)
    conv5 = layers.Conv2D(512, (3, 3),  padding = 'same', name ='conv5_2')(conv5)
    conv5 = layers.LeakyReLU()(conv5)
    conv5 = layers.BatchNormalization()(conv5)
    conv5 = layers.Conv2D(512, (3, 3),  padding = 'same', name ='conv5_3')(conv5)
    conv5 = layers.LeakyReLU()(conv5)
    conv5 = layers.BatchNormalization()(conv5)
    pool5 = layers.MaxPooling2D(pool_size = (2,2), strides = (2,2), name = 'pool_5')(conv5)

    #################################
    # Decoder
    #################################
    #conv1 = Conv2DTranspose(512, (2, 2), strides = 2, name = 'conv1')(pool5)

    upsp1 = layers.UpSampling2D(size = (2,2), name = 'upsp1')(pool5)
    conv6 = layers.Conv2D(512, 3,  padding = 'same', name = 'conv6_1')(upsp1)
    conv6 = layers.LeakyReLU()(conv6)
    conv6 = layers.BatchNormalization()(conv6)
    conv6 = layers.Conv2D(512, 3,  padding = 'same', name = 'conv6_2')(conv6)
    conv6 = layers.LeakyReLU()(conv6)
    conv6 = layers.BatchNormalization()(conv6)
    conv6 = layers.Conv2D(512, 3,  padding = 'same', name = 'conv6_3')(conv6)
    conv6 = layers.LeakyReLU()(conv6)
    conv6 = layers.BatchNormalization()(conv6)

    upsp2 = layers.UpSampling2D(size = (2,2), name = 'upsp2')(conv6)
    conv7 = layers.Conv2D(512, 3,  padding = 'same', name = 'conv7_1')(upsp2)
    conv7 = layers.LeakyReLU()(conv7)
    conv7 = layers.BatchNormalization()(conv7)
    conv7 = layers.Conv2D(512, 3,  padding = 'same', name = 'conv7_2')(conv7)
    conv7 = layers.LeakyReLU()(conv7)
    conv7 = layers.BatchNormalization()(conv7)
    conv7 = layers.Conv2D(512, 3,  padding = 'same', name = 'conv7_3')(conv7)
    conv7 = layers.LeakyReLU()(conv7)
    conv7 = layers.BatchNormalization()(conv7)
    #zero1 = layers.ZeroPadding2D(padding =  ((1, 0), (1, 0)), data_format = 'channels_last', name='zero1')(conv7)

    upsp3 = layers.UpSampling2D(size = (2,2), name = 'upsp3')(conv7)
    conv8 = layers.Conv2D(256, 3,  padding = 'same', name = 'conv8_1')(upsp3)
    conv8 = layers.LeakyReLU()(conv8)
    conv8 = layers.BatchNormalization()(conv8)
    conv8 = layers.Conv2D(256, 3,  padding = 'same', name = 'conv8_2')(conv8)
    conv8 = layers.LeakyReLU()(conv8)
    conv8 = layers.BatchNormalization()(conv8)
    conv8 = layers.Conv2D(256, 3,  padding = 'same', name = 'conv8_3')(conv8)
    conv8 = layers.LeakyReLU()(conv8)
    conv8 = layers.BatchNormalization()(conv8)

    upsp4 = layers.UpSampling2D(size = (2,2), name = 'upsp4')(conv8)
    conv9 = layers.Conv2D(128, 3,  padding = 'same', name = 'conv9_1')(upsp4)
    conv9 = layers.LeakyReLU()(conv9)
    conv9 = layers.BatchNormalization()(conv9)
    conv9 = layers.Conv2D(128, 3,  padding = 'same', name = 'conv9_2')(conv9)
    conv9 = layers.LeakyReLU()(conv9)
    conv9 = layers.BatchNormalization()(conv9)

    upsp5 = layers.UpSampling2D(size = (2,2), name = 'upsp5')(conv9)
    conv10 = layers.Conv2D(64, 3,  padding = 'same', name = 'conv10_1')(upsp5)
    conv10 = layers.LeakyReLU()(conv10)
    conv10 = layers.BatchNormalization()(conv10)
    conv10 = layers.Conv2D(64, 3,  padding = 'same', name = 'conv10_2')(conv10)
    conv10 = layers.LeakyReLU()(conv10)
    conv10 = layers.BatchNormalization()(conv10)

    conv11 = layers.Conv2D(3, 3, padding = 'same', name = 'conv11')(conv10)
    conv11 = keras.activations.sigmoid(conv11)

    model = keras.Model(inputs = inputs, outputs = conv11, name = 'vgg-16_encoder_decoder')

    return model

def train(train_gen, val_gen):

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
    strategy = tf.distribute.MirroredStrategy()
    print('Number of devices {}'.format(strategy.num_replicas_in_sync))

    with strategy.scope():
        model = create_model((224, 224, 3))
        model.compile(
            optimizer="adam", loss=[SSIMLoss], run_eagerly=True
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
    patient_data = pd.read_csv('./Seminoma_Outcomes_AnonSelection_20230124.csv', header=0)
    print(patient_data.head())

    
    pat_train, pat_val, y_train, y_val = train_test_split(
        patient_data['AnonPID'], patient_data['Meta'], test_size=0.25, random_state=42,
        stratify=patient_data['Meta']
    )

    print(list(pat_train))


    train_gen = MILdatagen(list(pat_train), 224, batch_size=64, train=True)
    val_gen = MILdatagen(list(pat_val), 224, batch_size=64, train=False)

    model = train(train_gen, val_gen)

    #encoder = keras.Model(model.input, model.layers[21].output)


    #train_gen = MILdatagen(list(pat_train), 224, batch_size=64, train=False)
    #val_gen = MILdatagen(list(pat_val), 224, batch_size=64, train=False)

    #X_train_embedded = encoder.predict(train_gen)
    #X_val_embedded = encoder.predict(val_gen)

    #np.save('./output/X_train_embedded.npy', X_train_embedded)
    #np.save('./output/X_val_embedded.npy', X_val_embedded)
