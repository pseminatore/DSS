from comet_ml import Experiment
import numpy as np
import pandas as pd
import librosa
import matplotlib.pyplot as plt
from scipy.io import wavfile as wav
from sklearn import metrics 
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split 
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, advanced_activations
from keras.optimizers import Adam, SGD
from keras.utils import to_categorical
import auth
import os

experiment = Experiment(api_key=auth.API_KEY,
                        project_name="DSS")

def main():
    features = []
    path = 'data/labeled_audio'
    list_subfolders_with_paths = [f.path for f in os.scandir(path) if f.is_dir()] 
    
    ## Grab folder associated with each class
    for folder in list_subfolders_with_paths:
        
        ## Make class folder current working dir
        cwd = os.path.join(os.getcwd(), folder)
        os.chdir(cwd)
        
        ## Iterate over sample in each folder
        for sample in os.listdir()[:10]:
            
            ## Show the name, then extract class label and actual data from folder path
            #print(sample)
            class_label = os.path.basename(os.path.normpath(folder))
            data = extract_features(sample)
            
            ## Log audio files to Comet for debugging
            #experiment.log_audio(sample, metadata = {'name': folder})
            
            ## Add to features list
            features.append([data, class_label])  
            
        ## Return to root directory
        os.chdir('..')
        os.chdir('..')
        os.chdir('..')
        
    ## Convert into a Pandas dataframe 
    featuresdf = pd.DataFrame(features, columns=['feature','class_label'])

    ## Convert features and corresponding classification labels into numpy arrays
    X = np.array(featuresdf.feature.tolist())
    y = np.array(featuresdf.class_label.tolist())
    
    ## Encode the classification labels
    le = LabelEncoder()
    yy = to_categorical(le.fit_transform(y))

    ## Split the dataset 
    x_train, x_test, y_train, y_test = train_test_split(X, yy, test_size=0.2, random_state = 127)
    
    ## Create model (not yet built)
    num_labels = yy.shape[1]
    filter_size = 2
    model = build_model(num_labels)
    
    ## perform fit on model
    num_epochs = 100
    num_batch_size = 32
    model.fit(x_train, y_train, batch_size=num_batch_size, epochs=num_epochs, validation_data=(x_test, y_test), verbose=1)
    model.summary()
    
    ## Evaluating the model on the training and testing set
    score = model.evaluate(x_train, y_train, verbose=0)
    print("Training Accuracy: {0:.2%}".format(score[1]))

    score = model.evaluate(x_test, y_test, verbose=0)
    print("Testing Accuracy: {0:.2%}".format(score[1]))
    
    print("")
    print("")
    print("")
    print("--------------------------------------------------------------------------------------------------------------------")

## Build simple FFNN (Feed Forward Neural Network) 
def build_model(num_labels, input_shape=(40,)):
    
    ## Create sequential model graph
    model = Sequential()
    
    ## TODO -- mess around with different activations and layer combinations
    ## Add layers
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dropout(0.3))
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dropout(0.3))
    model.add(Dense(num_labels))
    model.add(Activation('softmax'))
    
    ## For playing around with different optimizers
    opt = Adam(lr=0.01)
    
    ## Compile the model
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=opt)    
    
    return model

## Simple function to extract Mel Frequency Cepstral Coefficients (MFCCs) for every file in our dataset (taken from https://towardsdatascience.com/how-to-apply-machine-learning-and-deep-learning-methods-to-audio-analysis-615e286fcbbc)
def extract_features(file_name):
    audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast') 
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfccs_processed = np.mean(mfccs.T,axis=0)
     
    return mfccs_processed

if __name__ == '__main__':
    main()
    
