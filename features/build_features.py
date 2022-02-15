import os,sys
import tensorflow as tf
import numpy as np
import cv2
import random
import matplotlib.pyplot as plt

from conf import sr_config as config

class CustomDataGen(tf.keras.utils.Sequence):
    
    def __init__(self,
                 images_path = config.IMAGES,
                 labels_path = config.LABELS,
                 batch_size = config.BATCH_SIZE,
                 input_size=(config.INPUT_DIM, config.INPUT_DIM),
                 output_size = (config.LABEL_SIZE, config.LABEL_SIZE),
                 channel_size = 3,
                 shuffle=True):
                
        self.images_path = images_path
        self.labels_path = labels_path

        #labels and images has the same name
        self.names = os.listdir(images_path)
        
        self.batch_size = batch_size
        self.input_size = input_size
        self.output_size = output_size
        self.channel_size = channel_size
        self.shuffle = shuffle
        self.x_buffer = np.empty( (batch_size , input_size[0] , input_size[1], channel_size))
        self.y_buffer = np.empty( (batch_size , output_size[0] , output_size[1] , channel_size))
        self.n = len(self.names)
        
        if shuffle:
            random.shuffle(self.names)

    def on_epoch_end(self):  
        if shuffle:
            random.shuffle(self.names)
    
    def shuffleData(self):
        self.on_epoch_end()

    def __getitem__(self, index):
        
        for i in range(self.batch_size):
            self.x_buffer[i,:,:,:] = cv2.imread(os.path.join(self.images_path, self.names[index+i]))
            self.y_buffer[i,:,:,:] = cv2.imread(os.path.join(self.labels_path, self.names[index+i]))

        return self.x_buffer,self.y_buffer

    def __len__(self):
        return self.n // self.batch_size

    
