# -*- coding: utf-8 -*-
"""
Created on Wed Jan  6 01:20:45 2021

@author: yazılım
"""


"""

birleşik program

"""

get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import pandas as pd
import random
from scipy.io import wavfile
from sklearn.preprocessing import scale
import librosa.display
import librosa
import matplotlib.pyplot as plt
import os


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

from tensorflow.keras.models import load_model



sampling_rate=44100
sayac=1







"""

Spektogram kaydetmesi Tekil

"""
dosya=input("dosya adı girin: ")

# dosya konumunu yazın


dosya_konum="we/dataset/not/"+dosya+".wav"
data, sr = librosa.load(dosya_konum, sr=sampling_rate, mono=True)
data = scale(data)
        
melspec = librosa.feature.melspectrogram(y=data, sr=sr, n_mels=128)
           
log_melspec = librosa.power_to_db(melspec, ref=np.max)  
librosa.display.specshow(log_melspec, sr=sr)
            
# kayıt yolu yazın
directory = 'we/dataset'
if not os.path.exists(directory):
    os.makedirs(directory)
foto_ad="a"+str(sayac)    
plt.savefig(directory + '/' + (foto_ad) + '.png')

resim_yolu=(directory + '/' + (foto_ad) + '.png')

"""

ÖKSÜRÜK TESPİTİ

"""


# training oksuruk deil






image = tf.keras.preprocessing.image.load_img(resim_yolu)
giris = tf.keras.preprocessing.image.img_to_array(image)
giris = np.array([giris])  # Convert single image to a batch.





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
