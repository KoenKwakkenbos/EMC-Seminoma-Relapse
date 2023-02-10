import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from sklearn.model_selection import train_test_split, StratifiedKFold
import os
import random
import tensorflow as tf
import pandas as pd
import numpy as np
import cv2
from tqdm import tqdm
import tensorflow.keras.backend as K

from concurrent.futures import ThreadPoolExecutor

class MILdatagen(tf.keras.utils.Sequence):
    def __init__(self, slide_list, outcome_list, tile_size, batch_size=32, train=False):
        self.slide_list = slide_list
        self.pat_outcome_list = outcome_list
        self.tile_size = tile_size
        self.batch_size = batch_size
        self.train = train
        self.tile_list = []
        self.slide_tile_list = []
        self.tile_outcome_list = []

        for patient in self.slide_list:
            for root, subdirs, files in os.walk('/data/scratch/kkwakkenbos/Tiles_downsampled_1024/' + str(patient)):
                for file in files:
                    self.tile_list.append(os.path.join(root, file))
                    self.tile_outcome_list.append(self.pat_outcome_list[patient])
                    self.slide_tile_list.append(os.path.basename(root))

        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.indexes) / self.batch_size))

    def _process_image(self, tile):
        image = cv2.imread(tile)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_norm = self._normalize_image(image)
        #image_norm = image / 255.

        if self.train:
            image_norm = self._augment_image(image_norm)

        return image_norm

    def __getitem__(self, idx):
        indexes = self.indexes[idx*self.batch_size:(idx+1)*self.batch_size]

        with ThreadPoolExecutor(max_workers=32) as executor:
            X = list(executor.map(lambda tile: self._process_image(tile), [self.tile_list[k] for k in indexes]))

        X = np.array(X)
        y = np.array([self.tile_outcome_list[k] for k in indexes])

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

    def topk_dataset(self, idx):
        self.indexes = np.array(idx)
        if self.train:
            np.random.shuffle(self.indexes)

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.tile_list))
        #if self.train:
        #    np.random.shuffle(self.indexes)


patient_data = pd.read_csv('./Seminoma_Outcomes_AnonSelection_20230124.csv', header=0).set_index('AnonPID')

skf = StratifiedKFold(n_splits=3)
skf.get_n_splits(patient_data.index, patient_data['Meta'])

for i, (train_index, test_index) in enumerate(skf.split(patient_data.index, patient_data['Meta'])):
    K.clear_session()
    print(f"Fold {i}:")
    print(f"  Train: index={train_index}")
    print(f"  Test:  index={test_index}")

    print(patient_data.iloc[train_index])

    train_gen = MILdatagen(list(patient_data.iloc[train_index].index), patient_data['Meta'].iloc[train_index], 224, batch_size=32, train=True)
    val_gen = MILdatagen(list(patient_data.iloc[test_index].index), patient_data['Meta'].iloc[test_index], 224, batch_size=32, train=False)

    #inputs = keras.Input(shape=(224, 224, 3), name="digits")
    #x1 = ResNet50(include_top=False, weights='imagenet', pooling='max')(inputs)
    #outputs = layers.Dense(1, name="predictions")(x1)
    #model = keras.Model(inputs=inputs, outputs=outputs)
    #model.layers[1].trainable = False

    model = keras.Sequential([
        layers.BatchNormalization(input_shape=(224,224,3)),
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
        layers.Dense(1),
    ])
    print (model.summary())


    # Instantiate an optimizer.
    optimizer = keras.optimizers.Adam()
    # Instantiate a loss function.
    loss_fn = keras.losses.BinaryCrossentropy(from_logits=True)

    train_acc_metric = keras.metrics.BinaryAccuracy()
    val_acc_metric = keras.metrics.BinaryAccuracy()
    val_auc_metric = tf.keras.metrics.AUC()


    @tf.function
    def train_step(x, y):
        with tf.GradientTape() as tape:

            # Run the forward pass of the layer.
            # The operations that the layer applies
            # to its inputs are going to be recorded
            # on the GradientTape.
            logits = model(x, training=True)  # Logits for this minibatch
            
            # Compute the loss value for this minibatch.
            loss_value = loss_fn(y, logits)
        # Use the gradient tape to automatically retrieve
        # the gradients of the trainable variables with respect to the loss.
        grads = tape.gradient(loss_value, model.trainable_weights)

        # Run one step of gradient descent by updating
        # the value of the variables to minimize the loss.
        optimizer.apply_gradients(zip(grads, model.trainable_weights))
        train_acc_metric.update_state(y, logits)

        return loss_value

    @tf.function
    def inference_step(x):
        return model(x, training=False)

    @tf.function
    def test_step(x, y):
        val_logits = model(x, training=False)
        val_acc_metric.update_state(y, val_logits)
        val_auc_metric.update_state(y, val_logits)
        loss_value = loss_fn(y, tf.nn.sigmoid(val_logits))
        return loss_value

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1,
                                patience=10, min_lr=0.000001, verbose=1)

    early_stop = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=15, verbose=1)

    def group_argtopk(groups, data,k=5):
        data = data.numpy().ravel()
        order = np.lexsort((data, groups))
        groups = np.array(groups)
        groups = groups[order]
        data = data[order]
        index = np.empty(len(groups), 'bool')
        index[-k:] = True
        index[:-k] = groups[k:] != groups[:-k]
        return list(order[index])


    epochs = 200
    best_val_loss = np.Inf
    best_val_auc = -np.Inf

    reduce_lr.on_train_begin()
    early_stop.on_train_begin()

    for epoch in range(epochs):

        reduce_lr.on_epoch_begin(epoch)
        early_stop.on_epoch_begin(epoch)
        avg_loss = 0
        avg_loss_val = 0

        print("\nStart of epoch %d" % (epoch,))

        # Iterate over the batches of the dataset.
        logits = tf.zeros([0, 1])
        for step, (x_batch_train, y_batch_train) in enumerate(tqdm(train_gen)):
            logits_batch = inference_step(x_batch_train)
            logits = tf.concat([logits, logits_batch], 0)
        
        pats = train_gen.slide_tile_list[:min(len(train_gen.slide_tile_list), logits.shape[0])]
        topk_idx = group_argtopk(pats, logits, k=10)

        train_gen.topk_dataset(topk_idx)
        for step, (x_batch_train, y_batch_train) in enumerate(tqdm(train_gen)):
            loss_value = train_step(x_batch_train, y_batch_train)
            avg_loss += loss_value

        avg_loss /= (step+1)

        train_acc = train_acc_metric.result()
        print("Training acc over epoch: %.4f" % (float(train_acc),))
        print(
            "Training loss (for one batch) at step %d: %.4f"
            % (step, float(avg_loss))
        )

        # validation
        logits = tf.zeros([0, 1])
        for step, (x_batch_val, y_batch_val) in enumerate(tqdm(val_gen)):
            logits_batch = inference_step(x_batch_val)
            logits = tf.concat([logits, logits_batch], 0)
        
        pats = val_gen.slide_tile_list[:min(len(val_gen.slide_tile_list), logits.shape[0])]
        topk_idx = group_argtopk(pats, logits, k=10)

        val_gen.topk_dataset(topk_idx)

        for step, (x_batch_val, y_batch_val) in enumerate(tqdm(val_gen)):
            loss_value = test_step(x_batch_val, y_batch_val)
            avg_loss_val += loss_value
        avg_loss_val /= (step+1)
        if avg_loss_val < best_val_loss:
            model.save_weights(f"./output_MIL_kfold/best_model_weights_MILsmall_loss_fold{i+1}.h5")
            best_val_loss = avg_loss_val

        val_acc = val_acc_metric.result()
        val_auc = val_auc_metric.result()
        if val_auc > best_val_auc:
            model.save_weights(f"./output_MIL_kfold/best_model_weights_MILsmall_auc_fold{i+1}.h5")
            best_val_auc = val_auc

        print("Validation loss: %.4f" % (float(avg_loss_val),))
        print("Validation acc: %.4f" % (float(val_acc),))
        print("Validation auc: %.4f" % (float(val_auc),))

        train_acc_metric.reset_states()
        val_acc_metric.reset_states()
        val_auc_metric.reset_states()
        train_gen.on_epoch_end()
        val_gen.on_epoch_end()

    print(f'----------------------------- Fold {i+1} completed -----------------------------')