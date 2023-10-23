import time
import tensorflow as tf
import glob
import random
from tensorflow.keras.applications.resnet50 import preprocess_input
import numpy as np


def _process_image_augmentation(tile, clinical_vars, label):
    """
    Loads, normalizes, and applies data augmentation to an image tile.
    
    Parameters:
    - tile (str): The file path of the tile image to be processed.
    
    Returns:
    - A normalized and augmented image.
    """
    raw = tf.io.read_file(tile)
    image = tf.io.decode_image(raw)
    image = tf.cast(image, tf.float32)
    
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    image = tf.image.random_flip_up_down(image)
    image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
    image = tf.image.random_brightness(image, 0.2)
    
    return image, clinical_vars, label


def _process_image(tile, clinical_vars, label):
    """
    Loads, normalizes, and applies data augmentation to an image tile.
    
    Parameters:
    - tile (str): The file path of the tile image to be processed.
    
    Returns:
    - A normalized and augmented image.
    """
    raw = tf.io.read_file(tile)
    image = tf.io.decode_image(raw)
    image = tf.cast(image, tf.float32)

    return image, clinical_vars, label


def _normalize_image(tile, clinical_vars, label):

    image_norm = tile / 255
    
    return (image_norm, clinical_vars), label


def _normalize_image_imagenet(tile, clinical_vars, label):
    image_norm = preprocess_input(tile)
    
    return (image_norm, clinical_vars), label


def get_datagenerator(tile_paths, clinical_vars, labels, batch_size=4, train=True, imagenet=False):
    ds = tf.data.Dataset.from_tensor_slices((tile_paths,  clinical_vars, labels))
    ds = ds.shuffle(buffer_size=len(tile_paths),
                    reshuffle_each_iteration=True)
    
    if train:
        ds = ds.map(_process_image_augmentation,
                    num_parallel_calls=tf.data.experimental.AUTOTUNE)
    else:
        ds = ds.map(_process_image,
                    num_parallel_calls=tf.data.experimental.AUTOTUNE)

    if imagenet:
        ds = ds.map(_normalize_image_imagenet,
                    num_parallel_calls=tf.data.experimental.AUTOTUNE)
    else:
        ds = ds.map(_normalize_image,
                    num_parallel_calls=tf.data.experimental.AUTOTUNE)

    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.experimental.AUTOTUNE)

    return ds

if __name__ == "__main__":
    images = list(glob.glob("./tiles-TCGA-normalized/*/*.png"))
    clinical_vars = [[1, 2, 3] for i in range(len(images))]
    labels = [random.random() > 0.5 for i in range(len(images))]

    ds = get_datagenerator(images, clinical_vars, labels, batch_size=4, train=True, imagenet=False)
    for img, clin, label in ds.take(1):
        for im in img:
            plt.imshow(im)
            plt.show()
