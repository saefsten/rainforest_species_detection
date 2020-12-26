# rainforest_species_detection
Model for Kaggle challenge "Rainforest Connection Species Audio Detection"

The Kaggle challenge can be found here: https://www.kaggle.com/c/rfcx-species-audio-detection
In the challenge, around 2000 flac audio files are given with recordings from the rainforest. The file train_tp.csv contains information where in the files which species can be heard. Each file is 60 sec long and contains several species, both relevant and not relevant.

This model uses a Keras convolutional neural network to classify these species. The training data is created by randomly take samples of 200 ms from the training data, which are then labled samples. These are then converted to melspectograms using Librosa and these are used for training the model.

The predictions can then be made on the test data by looping over the test files and predict each 200 ms interval using the model. The final output is a probability for each species that this species is heard in the audio file. There is no determination of when in the file the species was heard.
