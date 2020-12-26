"""
Scrit can be used to cut the audio files and only save the labeled data in case the files are to big.
"""

import os
from tqdm import tqdm
import librosa
import pandas as pd
import soundfile as sf
import numpy as np

train_tp = pd.read_csv('PATH_TO_TRAIN CSV', sep=',')

def envelope(y, rate, threshold):
    mask = []
    y = pd.Series(y).apply(np.abs) # signal is sometimes negative
    y_mean = y.rolling(window=int(rate/10), min_periods=1, center=True).mean()
    for mean in y_mean:
        if mean > threshold:
            mask.append(True)
        else:
            mask.append(False)
    return mask

for item in tqdm(train_tp.index):
    f = train_tp.loc[item, 'recording_id']
    start = train_tp.loc[item, 't_min'] # seconds
    stop = train_tp.loc[item, 't_max'] # seconds
    duration = stop - start
    signal, sr = librosa.load('train.nosync/'+ f + '.flac', sr=48000, offset=start, duration=duration)
    species = train_tp.loc[item, 'species_id']
    filename = str(species) + '_' + str(f) + '.flac'
    sf.write('PATH_NEW_FILES'+filename, signal, sr, format='FLAC')
