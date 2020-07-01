#################################################################
## Author: Patrick Seminatore
## Created Date: 6/24/20
## Description: Utility functions for parsing NSYNTH dataset
##              from single .tar.gz file into organized folders 
##              by pitch
#################################################################

import tarfile
import json
import os
import shutil
from zipfile import ZipFile
import glob

pitch_scale = ['c', 'c#', 'd', 'd#', 'e', 'f', 'f#', 'g', 'g#', 'a', 'a#', 'b']

def main():
    #extract_data('nsynth-test.jsonwav')
    #organize_data()
    zip_audio()
    return 0


def extract_data(checkpoint_name, source_dir='data', target_dir='data/testing'):
    checkpoint_target = os.path.join(source_dir, f"{checkpoint_name}.tar.gz")
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    tar = tarfile.open(checkpoint_target)
    tar.extractall(target_dir)
    tar.close()
    
# CURRENTLY -- 'example' contains the string name of each training example
# TODO -- attatch name of example to pitch and store into correct folder  
def organize_data(source_dir='data/testing/nsynth-test', json_filename='examples', audio_dir='audio', target_dir = 'data/labeled_audio'):
    create_dirs(target_dir)
    json_file = open(os.path.join(source_dir, json_filename + '.json'))
    json_blob = json.load(json_file)
    for example in json_blob:
        midi_num = json_blob[example]['pitch']
        note = midiNum_to_pitch(midi_num)
        source_path = os.path.join(source_dir, audio_dir, example + '.wav')
        dest_path = os.path.join(target_dir, note, example + '.wav')
        shutil.move(source_path, dest_path)
    return 0


def zip_audio():
    path = 'data/labeled_audio'
    list_subfolders_with_paths = [f.path for f in os.scandir(path) if f.is_dir()] 
    for folder in list_subfolders_with_paths:
        cwd = os.path.join(os.getcwd(), folder)
        os.chdir(cwd)
        with ZipFile(os.getcwd() + '.zip', 'w') as zipper:
            for sample in os.listdir():
                zipper.write(sample)
        os.chdir('..')
        os.chdir('..')
        os.chdir('..')


def create_dirs(target_dir):
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    for note in pitch_scale:
        if not os.path.exists(os.path.join(target_dir, note)):
            os.makedirs(os.path.join(target_dir, note))    

# For ref - only uses sharp, no flat due to no way to make symbol
def midiNum_to_pitch(midi_num):
    pitch = (midi_num - 24) % 12
    return pitch_scale[pitch]

if __name__ == '__main__':
    main()
