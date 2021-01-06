# -*- coding: utf-8 -*-
"""
Created on Wed Jan  6 00:41:32 2021

@author: yazılım
"""


"""

ÖKSÜRÜK TESPİTİ

"""



from keras.callbacks import ModelCheckpoint, TensorBoard

import numpy as np 
import pandas as pd 


import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout
import time

import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array





# training oksuruk deil

#resim yolu girin

resim_yolu = 'we/dataset/a1.png'


image = tf.keras.preprocessing.image.load_img(resim_yolu)
giris = tf.keras.preprocessing.image.img_to_array(image)
giris = np.array([giris])  # Convert single image to a batch.
predictions3 = model.predict(giris)

predictions3


"""

Tahmin

"""


save_path = '.'
model2 = load_model(os.path.join(save_path,"bionluk.h5"))
pred = model2.predict(giris)
pred
for a in pred:
    print(a[1])
    sonuc=a[0]
    sonuc1=a[1]
    
if sonuc >0.50:
    print("SONUÇ ÖKSÜRÜK")
else:
    print("NEGATİF")