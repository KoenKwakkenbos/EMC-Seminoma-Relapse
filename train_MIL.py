import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from sklearn.model_selection import train_test_split
import os
import random
import tensorflow as tf
import pandas as pd
import numpy as np
import cv2
from tqdm import tqdm

from concurrent.futures import ThreadPoolExecutor

class MILdatagen(tf.keras.utils.Sequence):
    def __init__(self, pat_list, outcome_list, tile_size, batch_size =1, train=False):
        self.pat_list = pat_list
        self.slide_list = []
        self.pat_outcome_list = outcome_list
        self.tile_size = tile_size
        self.batch_size = batch_size
        self.train = train
        self.tile_list = []
        self.tile_outcome_list = []

        for pat in self.pat_list:
            for root, subdirs, files in os.walk('/data/scratch/kkwakkenbos/Tiles_downsampled_1024/' + str(pat)):
                for dir in subdirs:
                    self.slide_list.append((pat, dir))


    def __len__(self):
        return len(self.slide_list)

    def _process_image(self, tile):
        image = cv2.imread(os.path.join(tile))
        image_norm = self._normalize_image(image)

        #image_norm = image

        if self.train:
            image_norm = self._augment_image(image_norm)

        return image_norm

    def __getitem__(self, idx):
        tile_list = []
        for root, subdirs, files in os.walk('/data/scratch/kkwakkenbos/Tiles_downsampled_1024/' + str(self.slide_list[idx][0]) + '/' + str(self.slide_list[idx][1])):
            for file in files:
                tile_list.append(os.path.join(root, file))

        
        y = np.array(self.pat_outcome_list[self.slide_list[idx][0]])

        with ThreadPoolExecutor(max_workers=32) as executor:
            X = list(executor.map(self._process_image, tile_list))

        X = np.array(X)

        return X, y

    # Adapted from: https://github.com/schaugf/HEnorm_python
    def _normalize_image(self, img, Io=240, alpha=1, beta=0.15):
        ''' Normalize staining appearence of H&E stained images
        
        Example use:
            see test.py
            
        Input:
            I: RGB input image
            Io: (optional) transmitted light intensity
            
        Output:
            Inorm: normalized image
            H: hematoxylin image
            E: eosin image
        
        Reference: 
            A method for normalizing histology slides for quantitative analysis. M.
            Macenko et al., ISBI 2009
        '''

        HERef = np.array([[0.5626, 0.2159],
                        [0.7201, 0.8012],
                        [0.4062, 0.5581]])

        maxCRef = np.array([1.9705, 1.0308])

        # define height and width of image
        h, w, c = img.shape

        # reshape image
        img = img.reshape((-1,3))

        # calculate optical density
        OD = -np.log((img.astype(float)+1)/Io)

        # remove transparent pixels
        ODhat = OD[~np.any(OD<beta, axis=1)]

        # compute eigenvectors
        eigvals, eigvecs = np.linalg.eigh(np.cov(ODhat.T))

        #eigvecs *= -1

        #project on the plane spanned by the eigenvectors corresponding to the two 
        # largest eigenvalues    
        That = ODhat.dot(eigvecs[:,1:3])

        phi = np.arctan2(That[:,1],That[:,0])

        minPhi = np.percentile(phi, alpha)
        maxPhi = np.percentile(phi, 100-alpha)

        vMin = eigvecs[:,1:3].dot(np.array([(np.cos(minPhi), np.sin(minPhi))]).T)
        vMax = eigvecs[:,1:3].dot(np.array([(np.cos(maxPhi), np.sin(maxPhi))]).T)

        # a heuristic to make the vector corresponding to hematoxylin first and the 
        # one corresponding to eosin second
        if vMin[0] > vMax[0]:
            HE = np.array((vMin[:,0], vMax[:,0])).T
        else:
            HE = np.array((vMax[:,0], vMin[:,0])).T

        # rows correspond to channels (RGB), columns to OD values
        Y = np.reshape(OD, (-1, 3)).T

        # determine concentrations of the individual stains
        C = np.linalg.lstsq(HE,Y, rcond=None)[0]

        # normalize stain concentrations
        maxC = np.array([np.percentile(C[0,:], 99), np.percentile(C[1,:],99)])
        tmp = np.divide(maxC,maxCRef)
        C2 = np.divide(C,tmp[:, np.newaxis])

        # recreate the image using reference mixing matrix
        Inorm = np.multiply(Io, np.exp(-HERef.dot(C2)))
        Inorm[Inorm>255] = 254
        Inorm = np.reshape(Inorm.T, (h, w, 3)).astype(np.uint8)

        Inorm = cv2.normalize(Inorm, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

        return Inorm

    def _augment_image(self, img):

        if random.random() < 0.75:
            rot = random.choice([cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180, cv2.ROTATE_90_COUNTERCLOCKWISE])
            img = cv2.rotate(img, rot)

        if random.random() < 0.5:
            flip = random.choice([-1, 0, 1])
            img = cv2.flip(img, flip)

        return img

    def on_epoch_end(self):
        # shuffle the order of slides
        if self.shuffle:
            self._csv_slides.sample(frac=1).reset_index(drop=True)


patient_data = pd.read_csv('./Seminoma_Outcomes_AnonSelection_20230124.csv', header=0).set_index('AnonPID')
    
pat_train, pat_val, y_train, y_val = train_test_split(
    patient_data.index, patient_data['Meta'], test_size=0.25, random_state=42,
    stratify=patient_data['Meta']
)
train_gen = MILdatagen(list(pat_train), y_train, 224, batch_size=16, train=True)
val_gen = MILdatagen(list(pat_val), y_val, 224, batch_size=16, train=False)


inputs = keras.Input(shape=(224, 224, 3), name="digits")
inputs = preprocess_input(inputs)
x1 = ResNet50(include_top=False, weights='imagenet', pooling='max')(inputs)
outputs = layers.Dense(1, name="predictions")(x1)
model = keras.Model(inputs=inputs, outputs=outputs)
#model.layers[1].trainable = False

print(model.summary())

# Instantiate an optimizer.
optimizer = keras.optimizers.Adam()
# Instantiate a loss function.
loss_fn = keras.losses.BinaryCrossentropy(from_logits=True)

train_acc_metric = keras.metrics.BinaryAccuracy()
val_acc_metric = keras.metrics.BinaryAccuracy()


@tf.function
def train_step(x, y):    
    with tf.GradientTape() as tape:

        # Run the forward pass of the layer.
        # The operations that the layer applies
        # to its inputs are going to be recorded
        # on the GradientTape.
        logits = model(x_batch_train_k, training=True)  # Logits for this minibatch
        
        # Compute the loss value for this minibatch.
        y = [y] * 5
        y = tf.reshape(y, [5, 1])
        loss_value = loss_fn(y, logits, sample_weight=[0.7, 1.9])
    # Use the gradient tape to automatically retrieve
    # the gradients of the trainable variables with respect to the loss.
    grads = tape.gradient(loss_value, model.trainable_weights)

    # Run one step of gradient descent by updating
    # the value of the variables to minimize the loss.
    optimizer.apply_gradients(zip(grads, model.trainable_weights))
    train_acc_metric.update_state(y, logits)

    return loss_value

@tf.function
def test_step(x, y):
    y = [y] * 5
    y = tf.reshape(y, [5, 1])
    val_logits = model(x, training=False)
    val_acc_metric.update_state(y, val_logits)

    loss_value = loss_fn(y, val_logits, sample_weight=[0.7, 1.9])

    return loss_value

epochs = 20
for epoch in range(epochs):
    
    avg_loss = 0
    avg_loss_val = 0

    print("\nStart of epoch %d" % (epoch,))

    # Iterate over the batches of the dataset.
    for step, (x_batch_train, y_batch_train) in enumerate(tqdm(train_gen)):

        logits = model.predict(x_batch_train, 16, verbose=0)

        top_k = tf.math.top_k(tf.reshape(logits, [-1]), k=5, sorted=True).indices
        
        x_batch_train_k = tf.gather(x_batch_train, top_k)

        #x_batch_train = x_batch_train[0:10,]
        loss_value = train_step(x_batch_train_k, y_batch_train)
        avg_loss += loss_value
    
    avg_loss /= (step+1)
    
    train_acc = train_acc_metric.result()
    print("Training acc over epoch: %.4f" % (float(train_acc),))
    print(
        "Training loss (for one batch) at step %d: %.4f"
        % (step, float(avg_loss))
    )

    for step, (x_batch_val, y_batch_val) in enumerate(tqdm(val_gen)):
        logits = model.predict(x_batch_val, 16, verbose=0)

        top_k = tf.math.top_k(tf.reshape(logits, [-1]), k=5, sorted=True).indices
        
        x_batch_val_k = tf.gather(x_batch_val, top_k)

        loss_value = test_step(x_batch_val_k, y_batch_val)
        avg_loss_val += loss_value
    avg_loss_val /= (step+1)

    val_acc = val_acc_metric.result()
    print("Validation loss: %.4f" % (float(avg_loss_val),))
    print("Validation acc: %.4f" % (float(val_acc),))

    train_acc_metric.reset_states()
    val_acc_metric.reset_states()