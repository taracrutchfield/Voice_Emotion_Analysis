# standard packages
import json
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# machine learning
from sklearn.model_selection import train_test_split
from tensorflow import keras
from keras import preprocessing
from keras import layers
from keras import callbacks
import gc

def make_y(y_list,labels=None,gender_split='True'):
    new_y_list = []
    for y in y_list:
        if gender_split == 'False':
            y = y.str.split('_',expand=True)[0]

        if labels == None:
            labels = dict(zip(y.unique(), list(range(len(y.unique())))))
        y = y.replace(labels)
        y = keras.utils.to_categorical(y)
        new_y_list.append(y)
        
    return new_y_list, labels   

#--------------------------------------------------------------------------------------------------------------------

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
y = legend['emotion']+'_'+legend['sex']

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

    if config['final'] == 'False':
        # split into training and test sets
        X_train, X_test, y_train, y_test_og = train_test_split(X,y,train_size=.80,random_state=config['seed'])

        # get validation set from training set
        X_train, X_val, y_train, y_val = train_test_split(X_train,y_train,train_size=.80,random_state=config['seed'])
        new_y, labels = make_y([y_train,y_test_og,y_val],gender_split=config['split_gender'])
        y_train,y_test,y_val = new_y
        
    if config['final'] == 'True':
        # split into training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(X,y,train_size=.80,random_state=config['seed'])
        new_y, labels = make_y([y_train,y_val],gender_split=config['split_gender'])
        y_train,y_val = new_y
        
    # make a model
    model = keras.models.Sequential(name=config['model name'])

    # add empty layer with input shapes
    model.add(keras.Input(shape=X_train.shape[1:]))
    # add layers from json
    for key in config['layers'].keys():
        layer = config['layers'][key]
        if layer['layer'] == 'Conv2D':
            model.add(layers.Conv2D(layer['filters'],kernel_size=tuple(layer['kernel_size']),activation=layer['activation'],padding=layer['padding']))
        if layer['layer'] == 'MaxPooling2D':
            model.add(layers.MaxPooling2D(pool_size=tuple(layer['pool_size']),strides=layer['stride']))
        if layer['layer'] == 'AveragePooling2D':
            model.add(layers.AveragePooling2D(pool_size=tuple(layer['pool_size']),strides=layer['stride']))
        if layer['layer'] == 'GlobalAveragePooling2D':
            model.add(layers.GlobalAveragePooling2D())
        if layer['layer'] == 'GlobalMaxPool2D':
            model.add(layers.GlobalMaxPool2D())
        if layer['layer'] == 'Dropout':
            model.add(layers.Dropout(layer['rate'],seed=layer['seed']))
        if layer['layer'] == 'Dense':
            model.add(layers.Dense(layer['unit'],activation=layer['activation']))
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
    
    # callbacks
    class MyCustomCallback(callbacks.Callback):
          def on_epoch_end(self, epoch, logs=None):
                gc.collect()
                
    earlystopping = callbacks.EarlyStopping(monitor="val_loss", min_delta=0.005, patience=4,  
                                            restore_best_weights = True) 

    # fit model and print start and end times
    print('\nStart Time:',datetime.now())      
    history = model.fit(X_train,y_train,validation_data=(X_val,y_val),batch_size=100, 
                        epochs=config['epochs'], verbose=1, 
                        callbacks=[MyCustomCallback(),earlystopping])
    print('End Time:  ',datetime.now())
    
    # make a plot of the accuracy with each epoch
    EVA = plt.figure(figsize=(6,4),dpi=100)
    plot_name = 'EVA %s' % model.name
    plt.plot(history.history['accuracy'], label='accuracy',c='#4b3b75')
    plt.plot(history.history['val_accuracy'], label = 'validation accuracy',c='#91c64d')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title(model.name)
    plt.ylim(0,1)
    plt.legend()
    
    # save plot in Plots
    EVA.savefig('Data/Plots/'+plot_name);
    
    if config['final'] == 'True':
        model.save('Data/Model/'+str(model.name))
        model_config_path = 'Data/Model Config/Model_Labels_%s.json' % model.name
        with open(model_config_path , 'w') as fp:
            json.dump(labels, fp) 
        
    if config['final'] == 'False':
        predict = model.predict(X_test)

        predictions = pd.DataFrame(predict) 
        predictions.columns = labels.keys()
        predictions['true'] = y_test_og.values

        path = 'Data/CSVs/Predictions_%s.csv' % (model.name)
        predictions.to_csv(path)

    config_name = input("\nIf you wish to end the program please type 'quit'.\nIf you wish to test another model, please enter config json: ")
