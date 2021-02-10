# standard packages
import json
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# machine learning
from sklearn.model_selection import train_test_split
import keras
from keras import preprocessing
from keras import layers
from keras.callbacks import Callback
import gc

# import legend
legend = pd.read_csv('Data/CSVs/Audio Legend Clean.csv')[['emotion','filename','sex']].dropna()
legend['filename'] = legend['filename'].str.replace('.wav','',regex=True)

# set up X varible (Image Data)
X = []
print('\nLoading Images:')
for count, file in enumerate(legend['filename'],start=1):
    image = preprocessing.image.load_img('Data/Images/'+file+'.png')
    input_arr = preprocessing.image.img_to_array(image)
    X.append(input_arr)
    print('%4d of %d complete (%d%%)' % (count,len(legend['filename']),(count/len(legend['filename']))*100),end='\r')
print('\n')
X = np.array(X)

# Open JSON file 
config_name = input("Specify model config json file: ")

while config_name!= 'quit':
    break_statment = False
    while break_statment == False:
        if config_name == 'quit':
            break
        try: 
            with open('Data/Model Config/'+config_name) as json_file: 
                config = json.load(json_file) 
            break_statment = True
        except:
            config_name = input('Something went wrong, please make sure the file exists: ')
            
    # set up y variable
    if config['split_gender'] == 'True':
        y = legend['emotion']+'_'+legend['sex']
    if config['split_gender'] == 'False':
        y = legend['emotion']

    y = pd.factorize(y)
    
    labels = list(zip(list(range(len(y[1]))),y[1]))
    y = y[0]
    y = keras.utils.to_categorical(y)
   
    # split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X,y,train_size=.70,random_state=config['seed'])

    # split test set into a validation and test sets
    val_cutoff = int(len(X_test)/2)
    X_val = X_test[:val_cutoff]
    y_val = y_test[:val_cutoff]
    X_test = X_test[val_cutoff:]
    y_test = y_test[val_cutoff:]

    # make a model
    model = keras.models.Sequential(name=config['model name'])

    # add empty layer with input shapes
    model.add(keras.Input(shape=X_train.shape[1:]))
    # add layers from json
    for key in config['layers'].keys():
        layer = config['layers'][key]
        if layer['layer'] == 'Conv2D':
            model.add(layers.Conv2D(layer['filters'],
                                    kernel_size=tuple(layer['kernel_size']),
                                    activation=layer['activation'],
                                    padding=layer['padding']))
        if layer['layer'] == 'MaxPooling2D':
            model.add(layers.MaxPooling2D(pool_size=tuple(layer['pool_size']),
                                          strides=layer['stride']))
        if layer['layer'] == 'Dropout':
            model.add(layers.Dropout(layer['rate'],
                                     seed=layer['seed']))
        if layer['layer'] == 'Dense':
            model.add(layers.Dense(layer['unit'], 
                                   activation=layer['activation']))
        if layer['layer'] == 'Flatten':
            model.add(layers.Flatten())
    model.add(layers.Dense(len(labels),activation='softmax'))
    # compile
    compiler = config['compiler']
    model.compile(loss=compiler['loss'],
                  optimizer=compiler['optimizer'],
                  metrics=compiler['metrics'])
    # print summary    
    print(model.summary())

    class MyCustomCallback(Callback):
          def on_epoch_end(self, epoch, logs=None):
                gc.collect()

    # fit model and print start and end times
    print('\nStart Time:',datetime.now())      
    history = model.fit(X_train,y_train, validation_data=(X_val, y_val),batch_size=100, 
                        epochs=config['epochs'], verbose=1, callbacks=[MyCustomCallback()])
    print('End Time:  ',datetime.now())

    # make a plot of the accuracy with each epoch
    epoch_accuracy = plt.figure(figsize=(6,4),dpi=100)
    plot_name = 'Epoch v Accuracy - %s.png' % model.name
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title(model.name)
    plt.legend();

    # save plot in Plots
    epoch_accuracy.savefig('Data/Plots/'+plot_name)
    predict = model.predict(X_test)

    predictions = pd.DataFrame(predict).rename(columns=dict(labels))
    predictions['predict_int'] = [np.argmax(entry) for entry in predict]
    predictions['true_int'] = [np.argmax(entry) for entry in y_test]
    predictions['true_emo'] = predictions['true_int'].replace(dict(labels))

    accuracy = int(len(predictions[predictions['predict_int'] == predictions['true_int']])/len(predictions)*100)
    print('\nModel:',model.name)
    print('Accuracy: %d%%' % accuracy)

    path = 'Data/CSVs/Predictions_%s.csv' % (model.name)
    predictions.to_csv(path)

    config_name = input("\nIf you wish to end the program please type 'quit'.\nIf you wish to test another model, please enter config json: ")
