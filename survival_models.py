"""
File containing the three models corresponding to the three approaches used.

The classification and autoencoder models are both defined through the Keras
Model class.
The MIL model is defined as a Sequential model. This is because it was re-used from
a previous project. Within the scope of this current project, there is no difference in
the functioning of the three different model classes. 
All three models are defined in functions (create_[type]_model), that are called in the
training scripts. For the classification model, the parameter trainable can be used to
indicate whether the ResNet-50 model should have fixed weights during training.

Author: Koen Kwakkenbos
(k.kwakkenbos@student.tudelft.nl/k.kwakkenbos@gmail.com)
Version: 1.0
Date: 2022-02-17
"""

# imports
from tensorflow.keras import layers
from tensorflow.keras.applications import ResNet50
from tensorflow import keras

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

    x = layers.BatchNormalization()(image_input)
    resnet = ResNet50(input_shape=input_shape,
                      include_top=False,
                      weights='imagenet')
    resnet.trainable = trainable

    global_average_layer = layers.GlobalAveragePooling2D()

    x = resnet(x, training=trainable)
    x = global_average_layer(x)

    if num_clinical_features > 0:
        clin_in = layers.BatchNormalization()(clinical_input)
        concatenated_features = layers.concatenate([x, clin_in])

        y = layers.Dense(64, activation='relu')(concatenated_features)
    else:
        y = layers.Dense(64, activation='relu')(x)
    y = layers.Dropout(0.4)(y)
    # y = layers.Dense(32, activation='relu')(y)
    output = layers.Dense(1, activation='linear')(y)
    model = keras.Model(inputs=[image_input, clinical_input], outputs=output)

    model.layers[1].trainable = trainable

    return model


def create_cnn_model(input_shape=(512, 512, 3), num_clinical_features=3):
    input = layers.Input(shape=input_shape, name='image_input')
    clinical_input = layers.Input(shape=(num_clinical_features,), name='clinical_input')

    x = layers.BatchNormalization()(input)
    x = layers.Conv2D(64, kernel_size=(3, 3))(x)
    x = keras.layers.ReLU()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(64, kernel_size=(3, 3))(x)
    x = keras.layers.ReLU()(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    
    x = layers.Conv2D(128, kernel_size=(3, 3))(x)
    x = keras.layers.ReLU()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(128, kernel_size=(3, 3))(x)
    x = keras.layers.ReLU()(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    
    x = layers.Conv2D(256, kernel_size=(3, 3))(x)
    x = keras.layers.ReLU()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(256, kernel_size=(3, 3))(x)
    x = keras.layers.ReLU()(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    
    # x = layers.Conv2D(512, kernel_size=(3, 3))(x)
    # x = keras.layers.ReLU()(x)
    # x = layers.BatchNormalization()(x)
    # x = layers.Conv2D(512, kernel_size=(3, 3))(x)
    # x = keras.layers.ReLU()(x)
    # x = layers.BatchNormalization()(x)
    # x = layers.Conv2D(512, kernel_size=(3, 3))(x)
    # x = keras.layers.ReLU()(x)
    # x = layers.BatchNormalization()(x)
    # x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    
    image_output = layers.GlobalAveragePooling2D()(x)

    concatenated_features = layers.concatenate([image_output, clinical_input])

    y = layers.Dense(128, activation='relu')(concatenated_features)
    y = layers.Dropout(0.5)(y)
    y = layers.Dense(64, activation='relu')(y)
    y = layers.Dropout(0.5)(y)
    outputs = layers.Dense(1, activation='linear')(y)
    
    model = keras.Model(inputs=[input, clinical_input], outputs=outputs)
    return model

def create_small_cnn_model(input_shape=(224, 224, 3), num_clinical_features=3, trainable=False):
    """Creates a ResNet50-based binary classification model.

    Parameters:
    - input_shape (tuple, optional): The input shape of the model. Defaults to (224, 224, 3).
    - trainable (bool, optional): Whether the ResNet50 layers should be trainable. Defaults to False.

    Returns:
    - keras.Model: The ResNet50-based binary classification model.
    """

    image_input = layers.Input(input_shape, name='input')
    clinical_input = layers.Input(shape=(num_clinical_features,), name='clinical_input')

    # resnet = ResNet50(include_top=False, weights='imagenet', pooling='max')(image_input)
    
    x = layers.Conv2D(4 ,kernel_size=(3,3), activation='relu')(image_input)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(8 ,kernel_size=(3,3), activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(pool_size=((2,2)))(x)
    x = layers.Conv2D(16 ,kernel_size=(3,3), activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(16 ,kernel_size=(3,3), activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(pool_size=((2,2)))(x)
    x = layers.Conv2D(32 ,kernel_size=(3,3), activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(64 ,kernel_size=(3,3), activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling2D()(x)
    
    # x = layers.Dropout(0.3)(resnet)
    concatenated_features = layers.concatenate([x, clinical_input])
    y = layers.BatchNormalization()(concatenated_features)
    y = layers.Dropout(0.3)(y)
    y = layers.Dense(256, activation='relu')(y)
    y = layers.Dropout(0.3)(y)
    y = layers.Dense(128, activation='relu')(y)
    output = layers.Dense(1, activation='linear')(y)
    model = keras.Model(inputs=[image_input, clinical_input], outputs=output)

    # model.layers[1].trainable = trainable
    
    print(model.summary())

    return model


def create_MIL_model(input_shape=(224, 224, 3)):
    """
    Creates the classification model for the MIL approach.

    Parameters:
    - input_shape (tuple): A tuple specifying the input shape of the model. Defaults to (224, 224, 3).

    Returns:
    - keras.Model: A Keras convolutional model.

    """
    model = keras.Sequential([
            layers.BatchNormalization(input_shape=input_shape),
            layers.Conv2D(64 ,kernel_size=(3,3)),
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
            layers.Dense(10, activation='relu'),
            layers.Dense(5, activation='relu'),
            layers.Dense(1, activation='linear'),
        ])
    return model

def create_autoencoder_model(input_shape=(224, 224, 3)):
    """
    Creates a convolutional autoencoder model.

    Parameters:
    - input_shape (tuple): A tuple specifying the input shape of the model. Defaults to (224, 224, 3).

    Returns:
    - keras.Model: A Keras convolutional autoencoder model.

    """

    # Encoder part
    inputs = layers.Input(input_shape, name = 'input')

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

    # Decoder
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

    model = keras.Model(inputs=inputs, outputs=conv11, name = 'vgg-16_encoder_decoder')

    return model