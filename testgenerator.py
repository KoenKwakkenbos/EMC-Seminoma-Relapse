from datagenerator import MILdatagen

from tensorflow.keras.applications.resnet import ResNet50, preprocess_input

import sys

import tensorflow as tf
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from MIL_model import MILAttentionLayer, ReductionLayer

datagen = MILdatagen("./testcsv.csv", (224, 224), train=False, shuffle=False)


inp = layers.Input((224, 224, 3))
pre_process = preprocess_input(inp)
resnet = ResNet50(include_top=False, pooling='avg')(pre_process)
dense_1 = layers.Dense(128, activation="relu")(resnet)
dense_2 = layers.Dense(64, activation="relu")(dense_1)

# Invoke the attention layer.
alpha = MILAttentionLayer(
    weight_params_dim=256,
    kernel_regularizer=keras.regularizers.l2(0.01),
    use_gated=False,
    name="alpha",
)(dense_2)

# Multiply attention weights with the input layers.
multiply_layer = layers.multiply([alpha, dense_2])
reduction = ReductionLayer()(multiply_layer)
# Concatenate layers.
#concat = layers.concatenate(multiply_layer, axis=1)

# Classification output node.
output = layers.Dense(2, activation="softmax")(reduction)

model = keras.Model(inp, output)

print(model.summary())

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'], run_eagerly=True)
model.fit(datagen, verbose=1, epochs=10)

print("Fitting complete")