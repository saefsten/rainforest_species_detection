"""
Script that uses the saved model to predict autiofiles in a directory and saves them in csv file.
One probability is saved for each species per audio file.
"""

import pandas as pd
import numpy as np
import librosa
import os
from tqdm import tqdm
from tensorflow.keras.models import load_model

train = pd.read_csv('PATH_TO_TRAIN CSV FILE', sep=',')
fmax = train['f_max'].max()
fmin = train['f_min'].min()

class Settings:
    def __init__(self, hop_length=512, nfft=2048, sr=48000, fmax=fmax, fmin=fmin, length=0.2, power=1.5):
        self.hop_length = hop_length
        self.nfft = nfft
        self.sr = sr
        self.fmax = fmax
        self.fmin = fmin
        self.step = int(length * sr)
        self.power = power

settings = Settings()

model = load_model('PATH_TO_MODEL')
audiodir = 'PATH_TO_AUDIOFILES'

predictions = pd.DataFrame(columns=np.arange(0,24,1))

"""Loop through all files in directory and save predictions in the predictions dataframe"""
for f in os.listdir(audiodir):
    signal, sr = librosa.load(os.path.join(audiodir, f), sr=settings.sr)
    ypred = []
    recording = f[:-5] # skip the '.flac'
    for i in range(0, signal.shape[0]-settings.step, settings.step):
        sample = signal[i:i+settings.step]
        S = librosa.feature.melspectrogram(sample, sr=settings.sr, hop_length=settings.hop_length, 
            n_fft=settings.nfft, fmin=settings.fmin, fmax=settings.fmax, power=settings.power)
        X = S.reshape(1, S.shape[0], S.shape[1], 1)
        y_hat = model.predict(X)
        ypred.append(y_hat)
    
    pred = pd.DataFrame(ypred[0])
    for item in range(1,len(ypred)):
        new = pd.DataFrame(ypred[item])
        pred = pred.append(new,ignore_index=True)
    
    """ Add the average of the 3 highest prediction for each species as probability"""
    maxes = []
    for col in range(24):
        values = pred[col]
        top3average = values.sort_values(ascending=False).head(3).mean()
        average = round(top3average, 3)
        maxes.append(average)
    predictions.loc[recording] = maxes

predictions.to_csv('predictions.csv', float_format='%.3f')