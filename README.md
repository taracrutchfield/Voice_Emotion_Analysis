# Capstone 3 Project (WIP)

### Overview
  This project’s aim is to create a model that can determine the emotion present in a person’s voice. To do this I’m using data from both [CREMA-D](https://github.com/CheyneyComputerScience/CREMA-D) and [RAVDESS](https://smartlaboratory.org/ravdess/), two databases made up of short audio clips consisting of phrases spoken by actors trying to vocalize different emotion. Overall the data includes a total of 8877 voice clips portraying 8 different emotions: neutral, calm, happy, sad, anger, fear, disgust, and surprise. 

### Features
  Using the python package librosa, I’ve extracted numerous audio features from each clip. These features were normalized feature-wise, stacked, then saved as images. The features are as follows:

 - *mel-frequency spectrogram*: displays the spectrum of the sound.
 - *chroma_features*: describes the energy of each pitch class.
 - *spectral centroid*: the frequency that the spectrum is centered on, the "center of mass" of the sound.
 - *spectral bandwidth*: the width of the band at one-half the peak maximum.
 - *spectral roll off*: the shape of the signal.
 - *zero crossing rate*: measures the smoothness of the signal.
 - *mel-frequency cepstral coefficients*: describes the shape of the spectral envelope.

### Models
  Currently I’m testing different architectures for convolutional neural networks, the highest so far having an accuracy of 51% (random chance being around 14%). Depending on time, I would also like to look into using a LSTM recurrent neural network as well. 
