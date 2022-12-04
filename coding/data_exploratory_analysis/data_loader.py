import numpy as np
import pickle
from os import path
from tensorflow.keras.utils import to_categorical

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