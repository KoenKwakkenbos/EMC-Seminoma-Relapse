import os
import glob
import random
import tensorflow as tf
import numpy as np
import cv2
from concurrent.futures import ThreadPoolExecutor
from tensorflow.keras.applications.resnet50 import preprocess_input


def extract_identifier(filename):
    # Attempt extraction using both hyphen and underscore separators
    separators = ['-', '_']
    for separator in separators:
        parts = filename.split(separator)
        if len(parts) >= 3:
            identifier = separator.join(parts[:3])
            return identifier
    return None

def _make_riskset(time: np.ndarray) -> np.ndarray:
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
    return risk_set


class Datagen(tf.keras.utils.Sequence):
    """
    A generator class that yields batches of preprocessed image data and corresponding outcome
    labels.

    Parameters:
    - slide_list (list): A list of patient IDs corresponding to the slides from which tiles
      are to be extracted.
    - outcome_list (list): A list of binary outcome labels corresponding to the slide_list
      patients.
    - tile_size (int): The desired width and height in pixels of the square tiles to be extracted
      from each slide.
    - batch_size (int): The number of samples to generate per batch.
    - train (bool): A boolean indicating whether the generator is being used for training. If
      True, image augmentation will be applied to the generated batches.
    - imagenet (bool, optional): Whether to preprocess images for use with pre-trained Imagenet
      models. Default is False.
    """
    def __init__(self, df, tile_size, tile_path, batch_size=32, train=False, imagenet=False):
        self.df = df
        self.tile_size = tile_size
        self.tile_path = tile_path
        self.batch_size = batch_size
        self.train = train
        self.imagenet = imagenet
        self.minority_class_label = 1

        # Make the index three chunks long
        self.df.index = self.df.index.map(extract_identifier)

        # Precompute the list of all image paths
        self.tile_list = [os.path.join(subdir, file) 
                         for subdir, dirs, files in os.walk(self.tile_path)
                         for file in files if file.lower().endswith(('.png', '.jpg'))
                         and extract_identifier(file) in self.df.index]
        
        if self.train:
            # Perform oversampling

            labels = self.df.loc[[extract_identifier(os.path.basename(file)) for file in self.tile_list]]['Event'].to_list()
            class_counts = {label: labels.count(label) for label in set(labels)}
            print(f"Training --> oversampling. Class distribution before oversampling: {class_counts}")

            majority_class = 0
            minority_class = 1

            num_oversample_majority = int(class_counts[majority_class] * 0.20)
            num_oversample_minority = num_oversample_majority + (class_counts[majority_class] - class_counts[minority_class])

            majority_images = [img for img, label in zip(self.tile_list, labels) if label == majority_class]
            minority_images = [img for img, label in zip(self.tile_list, labels) if label == minority_class]

            oversampled_majority = np.random.choice(majority_images, num_oversample_majority, replace=True)
            oversampled_minority = np.random.choice(minority_images, num_oversample_minority, replace=True)

            self.tile_list.extend(oversampled_majority.tolist())
            self.tile_list.extend(oversampled_minority.tolist())

            class_counts[majority_class] += num_oversample_majority
            class_counts[minority_class] += num_oversample_minority
            print(f"Class distribution after oversampling: {class_counts}")

        print(self.tile_list[:5])

        self.on_epoch_end()

    def __len__(self):
        """
        Returns the number of batches that can be generated by this instance of the Datagen
        class.
        """
        return int(np.ceil(len(self.indexes) / self.batch_size))

    def _process_image(self, tile):
        """
        Loads, normalizes, and applies data augmentation to an image tile.
        
        Parameters:
        - tile (str): The file path of the tile image to be processed.
        
        Returns:
        - A normalized and augmented image.
        """
        image = cv2.imread(tile)
        image = image[:, :, ::-1].copy()

        # CHECK IF THIS IS REMOVED!!

        # image_norm = self._normalize_image(image)

        # ----------------------------------------

        
        if self.train:
            image = self._augment_image(image)
        
        if self.imagenet:
            image_norm = preprocess_input(image)
        else:
            image_norm = image / 255.
        
        return image_norm

    def __getitem__(self, idx):
        """
        Generates a batch of preprocessed image data and corresponding outcome labels.
        
        Parameters:
        - idx (int): The batch index.
        
        Returns:
        - A tuple containing a batch of image data and a batch of outcome labels.
        """
        indexes = self.indexes[idx*self.batch_size:(idx+1)*self.batch_size]
        to_process = [self.tile_list[k] for k in indexes]
        to_process = np.unique(to_process)
        # normalization removed, still necessary?
        with ThreadPoolExecutor() as executor:
            X_img = np.array(list(executor.map(lambda tile: self._process_image(tile), to_process)))

        # X_img = np.array([self._process_image(self.tile_list[k]) for k in indexes])
        
        batch_ids = [extract_identifier(os.path.basename(file)) for file in to_process]
        X_clin = np.array(self.df.loc[batch_ids][['LVI', 'RTI', 'Size', 'Treatment']])
        event = np.array(self.df.loc[batch_ids]['Event'])
        time = np.array(self.df.loc[batch_ids]['Time'])

        labels = {
            "label_event": event.astype(np.int32),
            "label_time": time.astype(np.float32),
            "label_riskset": _make_riskset(time)
        }

        return [X_img, X_clin], labels

    def _augment_image(self, img):
        """
        Applies random transformations to an image.

        Parameters:
        - image (numpy array): An image represented as a numpy array with shape (height, width, channels).

        Returns:
        - The augmented image as a numpy array with the same shape as the input image.
        """
        if random.random() < 0.75:
            rot = random.choice([cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180, cv2.ROTATE_90_COUNTERCLOCKWISE])
            img = cv2.rotate(img, rot)

        if random.random() < 0.75:
            flip = random.choice([-1, 0, 1])
            img = cv2.flip(img, flip)

        if random.random() < 75:
            img = cv2.convertScaleAbs(img, alpha=random.uniform(0.8, 1.2), beta=random.uniform(0.8, 1.2))

        return img

    def on_epoch_end(self):
        """
        Method called at the end of every epoch. Applies shuffling and resets the
        batch index.
        """
        self.indexes = np.arange(len(self.tile_list))
        if self.train:
            np.random.shuffle(self.indexes)

        self.__len__()


class DatagenMILSurv(Datagen):
    def __init__(self, df, tile_size, tile_path, batch_size=32, train=True, imagenet=False):
        # Call parent class's initialization
        super().__init__(df, tile_size, tile_path, batch_size, train, imagenet)
        self.slide_tile_list = np.array([extract_identifier(os.path.basename(file)) for file in self.tile_list])

    def topk_dataset(self, idx):
        """
        Set the indexes to be used for the next batch.

        Parameters:
        - idx (list): List of indexes to be used for the next training step (the top indexes).
        """

        # CHECK IF CORRECT:
        # If shuffeled, we need index of indexes

        self.indexes = self.indexes[np.array(idx)]
        if self.train:
            np.random.shuffle(self.indexes)

        self.__len__()

    # def on_epoch_end(self):
    #     """
    #     Reset the indexes back to the entire dataset for the next inference pass.
    #     """
    #     self.indexes = np.arange(len(self.tile_list))

        # Shuffle multiple lists the same way??
