import matplotlib.pyplot as plt
import random
import pickle
import numpy as np
from os import path
import os.path
from sklearn.model_selection import train_test_split
# from sklearn.externals 
import joblib
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import itertools
import pandas as pd
import time
from os import listdir
from shutil import copyfile
import pickle


from extract_audio_helper import *
from augmentation import augment_data,augment_background, convert_to_image

def execute_audio_extraction(audio_directory, audio_file_name, sample_rate, timestamp_directory,
                            number_seconds_to_extract, save_location):
    
    print ('Reading audio file (this can take some time)...')
    # Read in audio file
    librosa_audio, librosa_sample_rate = librosa.load(audio_directory+audio_file_name, 
                                                  sr=sample_rate)
    
    print ()
    print ('Reading done.')
    
    # Read gibbon labelled timestamp file
    gibbon_timestamps = read_and_process_gibbon_timestamps(timestamp_directory, 
                                   'g_'+audio_file_name[:audio_file_name.find('.wav')]+'.data', 
                                               sample_rate, sep=',')
    # Read non-gibbon labelled timestamp file
    non_gibbon_timestamps = read_and_process_nongibbon_timestamps(timestamp_directory,
                                   'n_'+audio_file_name[:audio_file_name.find('.wav')]+'.data', 
                                               librosa_sample_rate, sep=',')
    # Extract gibbon calls
    gibbon_extracted = extract_all_gibbon_calls(librosa_audio, gibbon_timestamps,
                                            number_seconds_to_extract,1, librosa_sample_rate,0)
    
    # Extract background noise
    noise_extracted = extract_all_nongibbon_calls(librosa_audio, non_gibbon_timestamps,
                                              number_seconds_to_extract,5, librosa_sample_rate,0)
    # Save the extracted data to disk
    pickle.dump(gibbon_extracted, open(save_location+'g_'+audio_file_name[:audio_file_name.find('.wav')]+'.pkl', "wb" ))
    pickle.dump(noise_extracted, open(save_location+'n_'+audio_file_name[:audio_file_name.find('.wav')]+'.pkl', "wb" )) 
    
    del librosa_audio
    print ()
    print ('Extracting segments done. Pickle files saved.')
    
    return gibbon_extracted,noise_extracted
    

def execute_augmentation(gibbon_extracted, 
                                  non_gibbon_extracted, number_seconds_to_extract, sample_rate,
                                  augmentation_amount_noise, augmentation_probability, 
                                  augmentation_amount_gibbon, seed, augment_directory, augment_image_directory,
                                  audio_file_name):
    
    print()
    print ('gibbon_extracted:',gibbon_extracted.shape)
    print ('non_gibbon_extracted:',non_gibbon_extracted.shape)
    
    non_gibbon_extracted_augmented = augment_background(seed, augmentation_amount_noise, 
                                                   augmentation_probability, non_gibbon_extracted, 
                                                   sample_rate, number_seconds_to_extract)
    
    gibbon_extracted_augmented = augment_data(seed, augmentation_amount_gibbon, 
                                              augmentation_probability, gibbon_extracted, 
                                              non_gibbon_extracted_augmented, sample_rate, 
                                              number_seconds_to_extract)
    

    
    sample_amount = gibbon_extracted_augmented.shape[0]
    
    non_gibbon_extracted_augmented = non_gibbon_extracted_augmented[np.random.choice(non_gibbon_extracted_augmented.shape[0], 
                                                                       sample_amount, 
                                                                       replace=True)]
    
    print()
    print('gibbon_extracted_augmented:',gibbon_extracted_augmented.shape)
    print('non_gibbon_extracted_augmented:',non_gibbon_extracted_augmented.shape)
    
    pickle.dump(gibbon_extracted_augmented, 
            open(augment_directory+'g_'+audio_file_name[:audio_file_name.find('.wav')]+'_augmented.pkl', "wb" ))

    pickle.dump(non_gibbon_extracted_augmented, 
                open(augment_directory+'n_'+audio_file_name[:audio_file_name.find('.wav')]+'_augmented.pkl', "wb" ))

    gibbon_extracted_augmented_image = convert_to_image(gibbon_extracted_augmented)
    non_gibbon_extracted_augmented_image = convert_to_image(non_gibbon_extracted_augmented)
    
    print()
    print ('gibbon_extracted_augmented_image:', gibbon_extracted_augmented_image.shape)
    print ('non_gibbon_extracted_augmented_image:', non_gibbon_extracted_augmented_image.shape)
    
    pickle.dump(gibbon_extracted_augmented_image, 
            open(augment_image_directory+'g_'+audio_file_name[:audio_file_name.find('.wav')]+'_augmented_img.pkl', "wb" ))

    pickle.dump(non_gibbon_extracted_augmented_image, 
                open(augment_image_directory+'n_'+audio_file_name[:audio_file_name.find('.wav')]+'_augmented_img.pkl', "wb" ))
    
    del non_gibbon_extracted_augmented, gibbon_extracted_augmented
    
    print()
    print ('Augmenting done. Pickle files saved to...')
    
    return gibbon_extracted_augmented_image, non_gibbon_extracted_augmented_image

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

def create_seed():
    return random.randint(1, 1000000)


def execute_preprocessing_all_files(training_file, audio_directory, 
                            sample_rate, timestamp_directory,
                            number_seconds_to_extract, save_location,
                            augmentation_amount_noise, augmentation_probability, 
                            augmentation_amount_gibbon, seed, augment_directory, augment_image_directory,
                            number_iterations):
    
    
    with open(training_file) as fp:
        line = fp.readline()

        while line:   
            file_name = line.strip()
            print ('Processing file: {}'.format(file_name))
            
            ## Extract segments from audio files
            gibbon_extracted, non_gibbon_extracted = execute_audio_extraction(audio_directory, 
                                         file_name, sample_rate, timestamp_directory,
                                         number_seconds_to_extract, save_location)
            
            ## Augment the extracted segments
            gibbon_extracted_augmented_image, non_gibbon_extracted_augmented_image = execute_augmentation(gibbon_extracted, 
                                  non_gibbon_extracted, number_seconds_to_extract, sample_rate,
                                  augmentation_amount_noise, augmentation_probability, 
                                  augmentation_amount_gibbon, seed, augment_directory, augment_image_directory,
                                  file_name)
            
            # Read next line
            line = fp.readline()

            
def load_training_images(training_folder, training_file):

    training_data = []
    gibbon_X = []
    noise_X = []
    with open(training_file) as fp:
        line = fp.readline()

        while line:

            file_name = line.strip()
            print()
            print('----------------------------------')
            print ('Reading file: {}'.format(file_name))
            file_name = file_name[:file_name.find('.wav')]

            if path.exists(training_folder+'g_'+file_name+'_augmented_img.pkl'):
                print ('Reading file gibbon augmented file: ', file_name)
                gibbon_X.extend(pickle.load(open(training_folder+'g_'+file_name+'_augmented_img.pkl', "rb" )))

            if path.exists(training_folder+'n_'+file_name+'_augmented_img.pkl'):
                print ('Reading non-gibbon augmented file:', file_name)
                noise_X.extend(pickle.load(open(training_folder+'n_'+file_name+'_augmented_img.pkl', "rb" )))

            # Read next line
            line = fp.readline()


    gibbon_X = np.asarray(gibbon_X)
    noise_X = np.asarray(noise_X)

    print()
    print ('Gibbon features:', gibbon_X.shape)
    print ('Non-gibbon features',noise_X.shape)
    
    return gibbon_X, noise_X

def prepare_X_and_Y(gibbon_X, noise_X):

    Y_gibbon = np.ones(len(gibbon_X))
    Y_noise = np.zeros(len(noise_X))
    X = np.concatenate([gibbon_X, noise_X])
    del gibbon_X, noise_X
    Y = np.concatenate([Y_gibbon, Y_noise])
    del Y_gibbon, Y_noise
    Y = to_categorical(Y)

    return X, Y