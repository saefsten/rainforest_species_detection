"""
Script that streams audio live from computer and predicts the probability and plots it in a bar plot.
"""

import numpy as np
import pyaudio
import librosa
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# for pyAudio
FORMAT = pyaudio.paFloat32
CHANNELS = 1
RATE = 48000
CHUNK = int(0.2 * RATE)
  
# for librosa:
hop_length = 512
nfft = 2048
sr = RATE
fmax = 13687.5
fmin = 93.75
step = CHUNK
power = 1.5
model = load_model('PATH_TO_MODEL')

# plotting
species = np.arange(0,24)
probs = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
axis_font = {'fontname':'Futura', 'size':'18'}


def callback(in_data, frame_count, time_info, flag):
    global probs
    in_data = np.frombuffer(in_data, dtype=np.float32)
    S = librosa.feature.melspectrogram(in_data, sr=sr, hop_length=hop_length, 
                                n_fft=nfft, fmin=fmin, fmax=fmax, power=power)
    X = S.reshape(1, S.shape[0], S.shape[1], 1)
    ypred = model.predict(X)
    probs = ypred[0]
    return (None, pyaudio.paContinue)

if __name__ == "__main__":
    p = pyaudio.PyAudio()
    
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    output=False,
                    stream_callback=callback,
                    frames_per_buffer=CHUNK,
                    input_device_index=0) # see 'testaudio.py' for how to check which index your audio has
    
    stream.start_stream()
    
    while True:
        def plot(i, species, probs):
            plt.cla()
            plt.bar(species, probs, color='#6C9B7D')
            plt.ylim(0, 1)
            plt.xticks(species)
            plt.xlabel('Species', fontdict=axis_font)
            plt.ylabel('Probability', fontdict=axis_font)
            ax = plt.axes()
            ax.set(facecolor='#EDFDF2')
        
        ani = FuncAnimation(plt.gcf(), plot, interval=200, fargs=(species, probs)) 
        plt.pause(0.0001)
        plt.tight_layout()