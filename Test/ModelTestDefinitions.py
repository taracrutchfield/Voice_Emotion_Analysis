# general packages
import os
import numpy as np
import pandas as pd
import json
# feature extraction
import librosa
import librosa.feature as lf
import matplotlib.pyplot as plt
# modeling
from tensorflow import keras
from keras import preprocessing

#------- Feature Extraction ------------------------------------------------------------------------------------
def audio_to_array(path):
    '''collects audio from a specified path and saves it as a dataframe'''
    files = os.listdir(path)
    audio = pd.DataFrame()
    for track in files:
        if '.wav' in track:
            try:
                clip, sample_rate = librosa.load(path+'/'+track, sr=16000)
            except:
                clip, sample_rate = np.nan,np.nan
        audio = audio.append({'track_name':track,'audio':clip,'sample_rate':sample_rate},ignore_index=True)
    audio = audio.set_index('track_name')
    return audio

def feature_extraction(audio):
    '''extracts features and saves them as a dataframe'''
    columns = ['track_name','mel spectrogram','chroma','mel cepstral']
    features = pd.DataFrame(columns=columns)
    for track in audio.index:
        clip = audio.loc[track]['audio']
        sample_rate = audio.loc[track]['sample_rate']
        try:
            L = [track,
                 # mel-frequency spectrogram
                 lf.melspectrogram(clip,sample_rate),
                 # chroma features
                 lf.chroma_stft(clip,sample_rate),        
                 # mel-frequency cepstral coefficients
                 lf.mfcc(clip,sample_rate)]
            features = features.append(dict(zip(columns,L)),ignore_index=True)
        except:
            print('An error has occured on the track "%s".' % (track))
    features = features.set_index('track_name')
    features['mel spectrogram'] = [librosa.power_to_db(entry, ref=np.max) for entry in features['mel spectrogram']]

    return features

def normalize(data,column):
    '''normalizes data in a column by a the appropriate scaler'''
    with open('../Data/Model Config/Scale_Config.json') as f:
        scalers = json.load(f)
    feat_min = scalers[column]['min']
    feat_max = scalers[column]['max']
    array = data[column].to_numpy()
    array = (array-feat_min)/(feat_max-feat_min)
    return array

def concat(data,columns):
    '''combines all the features into a single image'''
    concat_img = pd.DataFrame(columns=['track','final'])
    for index in data.index:
        for column in columns:
            entry = data.loc[index][column]
            if column == columns[0]:
                flat_entry = entry
            else:
                flat_entry = np.concatenate((flat_entry,entry),axis=0)
        concat_img = concat_img.append({'track':index,'final':flat_entry},ignore_index=True)
    return concat_img
    
def slice_and_pad(final):
    '''breaks long audio up and pads short audio such that the
    data is made up of 140 pixel long clips'''
    clips = pd.DataFrame()
    max_len=140
    for index in final.index:
        colname = index+'_slice'
        img = final.loc[index][0]
        array_len = img.shape[1]
        if array_len > max_len:
            n_slices = array_len//max_len
            remainder = array_len%max_len
            for n in range(n_slices):
                front = n*max_len
                back = front+max_len
                img_slice = img[:,front:back]
                clips = clips.append({colname:img_slice},ignore_index=True)
            if remainder != 0:
                img_slice = img[:,-remainder:]
                zeros = max_len - remainder
                front = int(zeros/2)
                back = int(zeros-front)
                img_slice = np.pad(img_slice, ((0,0),(front,back)))
                clips = clips.append({colname:img_slice},ignore_index=True)
        if array_len <= max_len:
            zeros = max_len - array_len
            front = int(zeros/2)
            back = int(zeros-front)
            img = np.pad(img, ((0,0),(front,back)))
            clips = clip.appends({colname:img},ignore_index=True)
    return clips

def save_clips(clips,img_path):
    '''saves clips in a specified folder'''
    for f in os.listdir(img_path):
        os.remove(img_path+'/'+f)
    for track in clips.columns:
        track_data = clips[track].dropna()
        for n,snippet in enumerate(track_data):
            path = img_path+'/%s__%d.png' % (track[:-10],n)
            plt.imsave(path,snippet)

def audio_to_images(audio_path,img_path):
    array = audio_to_array(audio_path)
    features = feature_extraction(array)

    # final dataset for normalized and stacked data
    final = pd.DataFrame()

    # normalizing data
    for column in features.columns:
        features[column] = normalize(features,column)

    # combining features
    final = concat(features,features.columns)
    final = final.set_index('track')

    # splitting images and padding small clips
    final = slice_and_pad(final)

    #saving images
    save_clips(final,img_path)

#------- Classification ------------------------------------------------------------------------------------

def load_images(folder_path):
    '''loads images in a given folder, returns the data as a 2D array (X)
    and the track name (y)'''
    X = []
    y = []
    folder = os.listdir(folder_path)
    for file in folder:
        image = preprocessing.image.load_img(folder_path+'/'+file)
        input_arr = preprocessing.image.img_to_array(image)
        X.append(input_arr)
        y.append(file.split('__')[0])
    X = np.array(X)
    return X,y

def get_predictions(X, model,labels):
    '''takes X and made a prediction using a specified model, returns 
    predictions as a dataframe.'''
    predictions = model.predict(X)
    predictions = pd.DataFrame(predictions) 
    predictions.columns = labels.keys()
    if len(labels) > 6:
        emotions = np.unique(np.array([emotion.split('_')[0] for emotion in labels.keys()]))
        for emotion in emotions:
            predictions[emotion] = (predictions[emotion+'_Male'] + predictions[emotion+'_Female'])/2
        predictions = predictions.drop(columns = labels.keys())
    return predictions

class predictions:
    '''class for the predictions'''
    def __init__(self,X,y):
        # map for the predicted lables
        with open('../Data/Model Config/Model_Labels_Avg_Pool_Split.json') as f:
            labels_AVG_SG = json.load(f)
        with open('../Data/Model Config/Model_Labels_Avg_Pool.json') as f:
            labels_AVG_EM = json.load(f)

        # load model
        AVG_SG = keras.models.load_model('../Data/Model/Avg_Pool_Split')
        AVG_EM = keras.models.load_model('../Data/Model/Avg_Pool')

        # organize predictions
        AVG_SG_pred = get_predictions(X, AVG_SG,labels_AVG_SG)
        AVG_EM_pred = get_predictions(X, AVG_EM,labels_AVG_EM)
        self.data = pd.DataFrame({'track':y})

        # get average
        for column in AVG_SG_pred.columns:
            self.data[column] = (AVG_SG_pred[column] + AVG_EM_pred[column])/2
        self.data = self.data.set_index('track')
        self.data['pred'] = self.data.idxmax(axis=1)
        
    def percent(self):
        '''returns a dataframe of what percentage of the clip was labeled with
        what emotion.'''
        percentage = pd.DataFrame(columns = self.data.columns[:-1])
        for track in self.data.index.unique():
            track_pred = self.data.loc[track]['pred']
            track_pred_percent = (track_pred.value_counts()/len(track_pred))*100
            track_pred_percent.name = track
            percentage = percentage.append(track_pred_percent)
        return percentage.fillna(0).astype(int)
        
    def probability(self):
        '''returns a dataframe of each emotion and the probability of that label
        being correct according to the model'''
        prob = self.data.loc[:]
        columns = prob.columns[:-1]
        prob['probability'] = prob[columns].max(axis=1)
        prob = prob.drop(columns=columns)
        return prob   