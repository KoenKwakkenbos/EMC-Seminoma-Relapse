import time
import tensorflow as tf
import glob
import random
from tensorflow.keras.applications.resnet50 import preprocess_input
import numpy as np


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


def _process_image_augmentation(tile, clinical_vars, labels_time, labels_event):
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
    
    return image, clinical_vars, labels_time, labels_event


def _process_image(tile, clinical_vars, labels_time, labels_event):
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

    return image, clinical_vars, labels_time, labels_event


def _normalize_image(tile, clinical_vars, labels_time, labels_event):

    image_norm = tile / 255
    
    return image_norm, clinical_vars, labels_time, labels_event


def _normalize_image_imagenet(tile, clinical_vars, labels_time, labels_event):
    image_norm = preprocess_input(tile)
    
    return image_norm, clinical_vars, labels_time, labels_event


def get_datagenerator(tile_paths, clinical_vars, labels_event, labels_time, batch_size, train, imagenet):
    ds = tf.data.Dataset.from_tensor_slices((tile_paths,  clinical_vars, labels_time, labels_event))
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
    ds = ds.map(lambda w, x, y, z: tf.numpy_function(_make_riskset, [w, x, y, z], [tf.float32, tf.float32, tf.float32, tf.int32, tf.bool]))
    ds = ds.map(_process_batch)
    ds = ds.prefetch(tf.data.experimental.AUTOTUNE)

    return ds

if __name__ == "__main__":
    images = list(glob.glob("./tiles_normalized/*/*.png"))
    clinical_vars = [[1, 2, 3] for i in range(len(images))]
    labels_event = [int(random.random() > 0.5) for i in range(len(images))]
    labels_time = [random.random() * 48 for i in range(len(images))]

    ds = get_datagenerator(images, clinical_vars, labels_event, labels_time, batch_size=4, train=True, imagenet=False)
    for img, clin, label in ds.take(1):
        for im in img:
            plt.imshow(im)
            plt.show()
