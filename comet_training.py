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
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import Adam
from keras.utils import to_categorical
import auth
import os

experiment = Experiment(api_key=auth.API_KEY,
                        project_name="DSS")

# Log audio files to Comet for debugging

path = 'data/labeled_audio'
list_subfolders_with_paths = [f.path for f in os.scandir(path) if f.is_dir()] 
for folder in list_subfolders_with_paths:
    cwd = os.path.join(os.getcwd(), folder)
    os.chdir(cwd)
    for sample in os.listdir():
        experiment.log_audio(sample, metadata = {'name': folder})    
    os.chdir('..')
    os.chdir('..')
    os.chdir('..')

    