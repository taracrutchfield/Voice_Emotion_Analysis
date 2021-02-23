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


def normalize(data,column):
    '''normalizes data in a column by a the appropriate scaler'''
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

#------- Feature Extraction ------------------------------------------------------------------------------------

# load the min and max of the training sets for normalizing 
with open('../Data/Model Config/Scale_Config.json') as f:
    scalers = json.load(f)

# collecting audio from folder
print('Collecting Clips')
path = 'Audio Clips Here!/'
files = os.listdir(path)
audio = pd.DataFrame()
for count, track in enumerate(files,start=1):
    if '.wav' in track:
        try:
            clip, sample_rate = librosa.load(path+track, sr=16000)
        except:
            clip, sample_rate = np.nan,np.nan
    audio = audio.append({'track_name':track,'audio':clip,'sample_rate':sample_rate},ignore_index=True)
    print('%d of %d complete (%d%%)' % (count,len(files),(count/len(files))*100),end='\r')   
audio = audio.set_index('track_name')
print('%-25s\n' % 'Complete!') 

# using librosa to get features
print("Extracting Features")
columns = ['track_name','mel spectrogram','chroma','mel cepstral']
features = pd.DataFrame(columns=columns)
for count,track in enumerate(audio.index,start=1):
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
    print('%d of %d complete (%d%%)' % (count,len(audio.index),(count/len(audio.index))*100)
          ,end='\r')
features = features.set_index('track_name')
features['mel spectrogram'] = [librosa.power_to_db(entry, ref=np.max) for entry in features['mel spectrogram']]
print('%-25s\n' % 'Complete!') 

# final dataset for normalized and stacked data
final = pd.DataFrame()

# normalizing data
print('Normalizing Features')
for column in features.columns:
    features[column] = normalize(features,column)
print('%-25s\n' % 'Complete!') 

# combining features
print('Combining Data')
final = concat(features,features.columns)
final = final.set_index('track')
print('%-25s\n' % 'Complete!') 

# splitting images and padding small clips
print('Splitting and Padding Audio')
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
print('%-25s\n' % 'Complete!') 

#saving images
print('Saving Images')
total_tracks, count = len(clips),1
for track in clips.columns:
    track_data = clips[track].dropna()
    for n,snippet in enumerate(track_data):
        path = 'Clip Images/%s__%d.png' % (track[:-10],n)
        plt.imsave(path,snippet)
        
        print('%4d of %d complete (%d%%)' % (count,total_tracks,(count/total_tracks)*100),end='\r')
        count+=1
print('%-30s' % ('Complete!')) 


#------- Classification ----------------------------------------------------------------------------------------

def get_predictions(model,labels):
    global X
    predictions = model.predict(X)
    predictions = pd.DataFrame(predictions) 
    predictions.columns = labels.keys()
    if len(labels) > 6:
        emotions = np.unique(np.array([emotion.split('_')[0] for emotion in labels.keys()]))
        for emotion in emotions:
            predictions[emotion] = (predictions[emotion+'_Male'] + predictions[emotion+'_Female'])/2
        predictions = predictions.drop(columns = labels.keys())
    return predictions

# loading images
X = []
y = []
folder = os.listdir('Clip Images')
for file in folder:
    image = preprocessing.image.load_img('Clip Images/'+file)
    input_arr = preprocessing.image.img_to_array(image)
    X.append(input_arr)
    y.append(file.split('__')[0])
X = np.array(X)

# map for the predicted lables
with open('../Data/Model Config/Model_Labels_Avg_Pool_Split.json') as f:
    labels_AVG_SG = json.load(f)
with open('../Data/Model Config/Model_Labels_Avg_Pool.json') as f:
    labels_AVG_EM = json.load(f)
    
# load model
AVG_SG = keras.models.load_model('../Data/Model/Avg_Pool_Split')
AVG_EM = keras.models.load_model('../Data/Model/Avg_Pool')

# organize predictions
AVG_SG_pred = get_predictions(AVG_SG,labels_AVG_SG)
AVG_EM_pred = get_predictions(AVG_EM,labels_AVG_EM)
predictions = pd.DataFrame({'track':y})

# get average
for column in AVG_SG_pred.columns:
    predictions[column] = (AVG_SG_pred[column] + AVG_EM_pred[column])/2
predictions = predictions.set_index('track')
predictions['pred'] = predictions.idxmax(axis=1)

# print predictions
for track in predictions.index.unique():
    track_pred = predictions.loc[track]['pred']
    track_pred_percent = (track_pred.value_counts()/len(track_pred))*100
    print('\nEmotions Detected in the Clip: %s' % track)
    for n in range(len(track_pred_percent)):
        print('%-7s  %2d%%' % (track_pred_percent.index[n],track_pred_percent[n]))

