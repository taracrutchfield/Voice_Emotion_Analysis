### Overview
   This project’s aim was to train a convolutional neural network to recognize emotions in a voice. Data was collected from two databases [CREMA-D](https://github.com/CheyneyComputerScience/CREMA-D) and [RAVDESS](https://smartlaboratory.org/ravdess/), which both included short clips of actor vocalized phrases. In total, 8651 audio clips covering 6 separate emotions, *Neutral*, *Happy*, *Sad*, *Angry*, *Fear*, and *Disgust*, were collected for this project. 

### Features
   In order to train a model to detect emotions in sound, features had to be extracted from each clip and processed for modeling. Using the python package librosa, three audio features were extracted from the data, each presented as a 2D image with time as the x-axis. These features were then normalized feature-wise, stacked, and padded such that the final modeling data was made up of 160x140 pixel images. 

Features Description:
 - **mel-frequency spectrogram**: displays the spectrum of sound based on how humans recognize shifts in  frequency. 
 - **mel-frequency cepstral coefficients**: describes the shape of the spectral envelope and provided inside on the shape of the vocal tract.
 - **chroma features**: portrays the energy at each pitch class.

### Models
   Two separate architectures and two kinds of training sets were experimented with, combining for a total of four models. The two architectures were both composed of repeating 2D convolution layers + pooling layers, the main difference between them being the kind of pooling layers used (average v max). Of the two training sets, one was made up of the six emotion labels alone while the other included both emotion and actor gender for a total of 12 labels.

### Conclusion
   Overall, the average pooling models did the best, with both the emotion and gender+emotion labels fairing equally well. Due to being the top two models, and the fact that their independent emotion scores were complimentary, the results of the two average pooling models were combined to create the final model, with a testing accuracy of 59% (random change being 17%).
   Currently I’m testing this final model using emotional scenes from famous movies. Here are the results as of this last update:

- [A Few Good Men](https://youtu.be/9FnO3igOkOk?t=39) (Anger/Disgust Test)
    - Anger    47%
    - Disgust  36%
    - Sad      10%
    - Other     7%
    
- [Steel Magnolias](https://www.youtube.com/watch?v=iZx1W6cHw-g) (Sad Test)
    - Sad     80%
    - Fear     10%
    - Other    10%
    
- [Pulp Fiction](https://youtu.be/qo5jnBJvGUs?t=26) (Anger Test)
    - Anger   100%
    
- [The Blair Witch Project](https://www.youtube.com/watch?v=2m_lqGnLtWA) (Fear/Sad Test)
    - Fear     50%
    - Sad      32%
    - Other    13%
    
- [Elf](https://www.youtube.com/watch?v=fNMtHosai08) (Happy/Neutral)
    - Sad      51%
    - Fear     33%
    - Other    13%