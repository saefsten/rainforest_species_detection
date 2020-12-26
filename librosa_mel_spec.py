"""
First a the samples are created from the training data. The number of samples is set to 18.000 with 20% saved for validation.
"""

import pandas as pd
import numpy as np
import librosa
import warnings
from tqdm import tqdm
from tensorflow.keras.utils import to_categorical
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras import Sequential
from tensorflow.keras.metrics import categorical_accuracy, categorical_crossentropy
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.callbacks import EarlyStopping

warnings.simplefilter('ignore', category=UserWarning) # to ignore user warning when reading flac files

train = pd.read_csv('PATH_TO_TRAIN_CSV', sep=',')
fmax = train['f_max'].max()
fmin = train['f_min'].min()
class_dist = train.groupby(['species_id'])['species_id'].mean()


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

def create_samples(n_samples):
    """Creates samples of 200 ms length that are randomly picked. A random class is first picked, followed by a random pick of a audiofile as well as a slice of that file."""
    X = []
    y = []
    for _ in tqdm(range(n_samples)):
        rand_class = np.random.choice(class_dist.index) # select a random class
        recording = np.random.choice(train[train['species_id']==rand_class]['recording_id']) # random choice from list of files with this class
        species_id = train[train['recording_id']==recording].iloc[0,1]
        signal, sr = librosa.load('train_cut_48k.nosync/'+str(species_id)+'_'+str(recording)+'.flac', sr=settings.sr)
        stop_index = signal.shape[0] - settings.step
        rand_index = np.random.randint(0, stop_index)
        sample = signal[rand_index : rand_index + settings.step] # random part of the whole time interval of length step
        S = librosa.feature.melspectrogram(sample, sr=settings.sr, hop_length=settings.hop_length, n_fft=settings.nfft, fmin=settings.fmin, fmax=settings.fmax, power=settings.power)
        X.append(S)
        y.append(species_id)
  
    X, y = np.array(X), np.array(y)
    X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)
    y = to_categorical(y, num_classes=24)
    return X, y

def make_cnn(input_shape):
    """Creates a Keras convolutional neural network"""
    model = Sequential()
    model.add(Conv2D(10, kernel_size=(3, 3), activation='relu', strides=(1, 1), padding='same', input_shape=input_shape))
    model.add(Conv2D(14, kernel_size=(3, 3), activation='relu', strides=(1, 1), padding='same'))
    model.add(MaxPooling2D(2, 2))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(12, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(24, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    return model

X, y = create_samples(18000)
print('X:', X.shape)

input_shape = (X.shape[1], X.shape[2], X.shape[3])

model = make_cnn(input_shape)

callback = EarlyStopping(monitor='val_accuracy', patience=2)
model.fit(X, y, epochs=30, batch_size=800, validation_split=0.2, callbacks=[callback])
model.save('MODEL_NAME')