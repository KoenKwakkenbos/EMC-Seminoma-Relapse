import os
import random
import tensorflow as tf
import pandas as pd
import numpy as np
import cv2
from tqdm import tqdm

from concurrent.futures import ThreadPoolExecutor, as_completed

class MILdatagen(tf.keras.utils.Sequence):
    def __init__(self, slide_list, outcome_list, tile_size, batch_size =32, train=False):
        self.slide_list = slide_list
        self.pat_outcome_list = outcome_list
        self.tile_size = tile_size
        self.batch_size = batch_size
        self.train = train
        self.tile_list = []
        self.tile_outcome_list = []

        for patient in self.slide_list:
            for root, subdirs, files in os.walk('/data/scratch/kkwakkenbos/Tiles_downsampled/' + str(patient)):
                for file in files:
                    self.tile_list.append(os.path.join(root, file))
                    self.tile_outcome_list.append(self.pat_outcome_list[patient])

        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.tile_list) / self.batch_size))

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

        #X = np.empty((self.batch_size, *self.tile_size))

        #for i, file in enumerate([self.tile_list[k] for k in indexes]):
        #    X[i,] = self._process_image(file)
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
        OD = -np.log((img.astype(np.float)+1)/Io)
        
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
        self.indexes = np.arange(len(self.tile_list))
        if self.train:
            np.random.shuffle(self.indexes)
