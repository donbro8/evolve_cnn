import os
import yaml
import shutil
import wget
import numpy as np
import logging
import pandas
from zipfile import ZipFile
from data_preprocessing.preprocess_helper import *
from gblock.functions import initiate_logger, load_yaml
from keras.datasets import mnist, cifar10, cifar100



def extract_zip_files(zip_file_path, extract_path):

    files = os.listdir(zip_file_path)

    n_zip_files = 0

    for file in files:

        destination_path = extract_path + file[:file.find('.zip')].lower()
        
        if '.zip' in file:

            n_zip_files += 1

            with ZipFile(zip_file_path + file, 'r') as zipObj:

                logging.info(f'Extracting {file} to {destination_path}')

                print(f'Extracting {file} to {destination_path}')

                zipObj.extractall(destination_path)

        for (dir_path, _, file_names) in os.walk(extract_path + file[:file.find('.zip')].lower()):

            if len(file_names) > 0 and dir_path != destination_path:

                for file_name in file_names:

                    logging.info(f'Copying {file_name} from {dir_path} to {destination_path}')

                    print(f'Copying {file_name} from {dir_path} to {destination_path}')

                    shutil.copy(dir_path + '/' + file_name, destination_path)

                shutil.rmtree(dir_path)

    if n_zip_files == 0:

        logging.info(f'No zip files found in directory {zip_file_path}')

        raise Exception('No zip files found in directory')
    



def fetch_raw_data(source_path, destination_path, expected_file_names = None):

    """
    Copies files from source directory to destination directory
    """

    if not os.path.exists(source_path):
        
        logging.info(f'Source directory {source_path} does not exist')

        raise Exception('Source directory does not exist')
    
    file_names = os.listdir(source_path)

    if len(set(file_names) - set(expected_file_names)) == len(file_names):

        logging.info(f'No expected files found in source directory. Trying to fetch raw data from {source_path}...')

        raise Exception('No expected files found in source directory')
    
    else:

        copied_files = []
        uncopied_files = []

        for file in expected_file_names:

            try:

                shutil.copy(source_path + '/' + file, destination_path)
                copied_files.append(file)

            except:
                
                uncopied_files.append(file)

        if len(uncopied_files) > 0:

            logging.info(f'The following files were not copied: {uncopied_files}')

            print(f'The following files were not copied: {uncopied_files}')

        elif set(copied_files) == set(expected_file_names):

            logging.info('All files were copied successfully')

            print('All files were copied successfully')




def compile_text_files(directory, destination, file_name):
    """
    Compiles all file names in a directory into one text file
    """

    file_names = os.listdir(directory)

    logging.info(f'Compiling {len(file_names)} files into {file_name}')

    with open(destination + '/' + file_name, 'w') as f:

        f.writelines('\n'.join(file_names))



def move_extra_files(file_names, directory):

    """
    Move files if they are not in the list of file names
    """

    dir_files = os.listdir(directory)

    extra_file_paths = []

    for file in dir_files:

        file_name = file.split('.')[0]

        if file_name[file_name.find('H'):] not in file_names:

            extra_file_paths.append(directory + '/' + file)

    if len(extra_file_paths) > 0:

        os.makedirs(directory + '/extra_files', exist_ok = True)

        for path in extra_file_paths:

            parent_paths = directory.split('/')

            shutil.move(path, '/'.join(parent_paths[:-1]) + f'/{parent_paths[-1]}_extra_files')



def load_gibbon_data(path_to_raw_gibbon_data = None, download = False, overwrite = False):

    # Get current working directory
    cwd = os.getcwd()

    data_dir = cwd + '/data/'

    raw_data_dir = data_dir + 'raw_gibbon_data/'

    initiate_logger(data_dir + 'gibbon_load_log.txt')

    try:
        # Load preprocessed gibbon data
        X_train = np.load(data_dir + 'processed_data/gibbon/X_train.npy')
        X_test = np.load(data_dir + 'processed_data/gibbon/X_test.npy')
        Y_train = np.load(data_dir + 'processed_data/gibbon/Y_train.npy')
        Y_test = np.load(data_dir + 'processed_data/gibbon/Y_test.npy')

        logging.info('Preprocessed gibbon data loaded successfully')
        print('Preprocessed gibbon data loaded successfully')

    except:

        logging.info('Preprocessed gibbon data not found. Trying to preprocess raw data...')
        print('Preprocessed gibbon data not found. Trying to preprocess raw data...')

        if not os.path.exists(raw_data_dir) or overwrite:

            files = ['Test.zip', 'Test_Labels.zip', 'Train.zip', 'Train_Labels.zip']

            try:
                # Unzip raw data from data/raw_gibbon_data
                extract_zip_files(raw_data_dir, raw_data_dir)

            except:

                logging.info(f'Raw gibbon data not found in {raw_data_dir}. Trying to fetch raw data...')

                print(f'Raw gibbon data not found in {raw_data_dir}. Trying to fetch raw data...')

                if overwrite:

                    logging.info('Overwrite set to True. Deleting existing raw data directory...')

                    print('Overwrite set to True. Deleting existing raw data directory...')

                    shutil.rmtree(raw_data_dir)
                
                os.makedirs(raw_data_dir)

                # Check if path to raw data is provided and if all the files are there
                if path_to_raw_gibbon_data != None:

                    logging.info(f'Trying to fetch raw gibbon data from {path_to_raw_gibbon_data}...')

                    print(f"Trying to fetch raw gibbon data from {path_to_raw_gibbon_data}...")

                    fetch_raw_data(path_to_raw_gibbon_data, raw_data_dir, files)
                    extract_zip_files(raw_data_dir, raw_data_dir)

                else:

                    logging.info('No path to raw gibbon data provided.')

                    print('No path to raw gibbon data provided.')

                    if download:

                        logging.info('Trying to download raw gibbon data...')

                        print('Trying to download raw gibbon data...')

                        
                        gibbon_urls = ['https://zenodo.org/record/3991714/files/Test.zip?download=1', 'https://zenodo.org/record/3991714/files/Test_Labels.zip?download=1',
                                        'https://zenodo.org/record/3991714/files/Train.zip?download=1', 'https://zenodo.org/record/3991714/files/Train_Labels.zip?download=1']

                        missing_files = list(set(files) - set(os.listdir(raw_data_dir)))
                        
                        for url in gibbon_urls:

                            for file in missing_files:

                                if file in url:

                                    logging.info(f'Downloading {file} from {url}...')

                                    print(f'Downloading {file} from {url}...')
                                
                                    wget.download(url, raw_data_dir)

                        extract_zip_files(raw_data_dir, raw_data_dir)

                    else:

                        logging.info('No path to raw gibbon data provided and download set to False.')

                        raise Exception('No path to raw gibbon data provided and download set to False.')
                    
        
        file_names = load_yaml(data_dir + 'file_names.yaml')
                
        # Set parameters for preprocessing (can think about moving this to a config file)
        sample_rate = 4800
        number_seconds_to_extract = 10
        seed = 42
        number_iterations = 1
        augmentation_probability = 1.0
        augmentation_amount_noise = 2
        augmentation_amount_gibbon = 10

        try:
            saved_data = list(set([x.replace('.npy', '').split('_')[-1] for x in os.listdir(data_dir + 'processed_data/gibbon/')]))
            data_types_missing = [x for x in ['train', 'test'] if x not in saved_data]

        except:
            os.makedirs(data_dir + 'processed_data/gibbon/', exist_ok=True)
            data_types_missing = ['train', 'test']

        for data_type in data_types_missing:

            logging.info(f'Executing preprocessing for {data_type} data...')

            print('Executing preprocessing for ' + data_type + ' data...')

            move_extra_files(file_names[data_type], raw_data_dir + data_type + '/', )
        
            compile_text_files(raw_data_dir + data_type, raw_data_dir, f'{data_type}.txt')

            audio_directory = raw_data_dir + data_type + '/'
            timestamp_directory = raw_data_dir + data_type + '_labels/'
            save_location = raw_data_dir + data_type + '_pickled_data/'
            augment_directory = raw_data_dir + data_type + '_augmented_data/'
            augment_image_directory = raw_data_dir + data_type + '_augmented_image_data/'
            file_names = raw_data_dir + f'{data_type}.txt'

            os.makedirs(save_location, exist_ok=True)
            os.makedirs(augment_directory, exist_ok=True)
            os.makedirs(augment_image_directory, exist_ok=True)

            execute_preprocessing_all_files(file_names, audio_directory, 
                        sample_rate, timestamp_directory,
                        number_seconds_to_extract, save_location,
                        augmentation_amount_noise, augmentation_probability, 
                        augmentation_amount_gibbon, seed, augment_directory, augment_image_directory,
                        number_iterations)
            
            logging.info(f'Preprocessing for {data_type} data complete. Saving data...')
            
            print('Preprocessing for ' + data_type + ' data complete. Saving data...')
            
            # Load the gibbon and noise data
            g_X, n_X = load_training_images(augment_image_directory, file_names)

            # Update data format
            X, Y = prepare_X_and_Y(g_X, n_X)

            # Save the data data_dir + 'processed_data/gibbon/X_train.npy'
            np.save(data_dir + f'processed_data/gibbon/X_{data_type}.npy', X)
            np.save(data_dir + f'processed_data/gibbon/Y_{data_type}.npy', Y)


        # Load preprocessed gibbon data
        X_train = np.load(data_dir + 'processed_data/gibbon/X_train.npy')
        X_test = np.load(data_dir + 'processed_data/gibbon/X_test.npy')
        Y_train = np.load(data_dir + 'processed_data/gibbon/Y_train.npy')
        Y_test = np.load(data_dir + 'processed_data/gibbon/Y_test.npy')

        logging.info('Preprocessed gibbon data loaded successfully')
        
        print('Preprocessed gibbon data loaded successfully')
            
    return (X_train, Y_train), (X_test, Y_test)


def load_mnist_data():

    # Get current working directory
    cwd = os.getcwd()

    data_dir = cwd + '/data/'

    initiate_logger(data_dir + 'mnist_load_log.txt')

    try:
        # Load preprocessed mnist data
        X_train = np.load(data_dir + 'processed_data/mnist/X_train.npy')
        X_test = np.load(data_dir + 'processed_data/mnist/X_test.npy')
        Y_train = np.load(data_dir + 'processed_data/mnist/Y_train.npy')
        Y_test = np.load(data_dir + 'processed_data/mnist/Y_test.npy')

        logging.info('Preprocessed mnist data loaded successfully')

        print('Preprocessed mnist data loaded successfully')

    except:

        logging.info('Preprocessed mnist data not found. Trying to download data...')

        print('Preprocessed mnist data not found. Trying to download data...')

        os.makedirs(data_dir + 'processed_data/mnist/')

        # Load raw mnist data
        (X_train, Y_train), (X_test, Y_test) = mnist.load_data()

        # Save the data
        np.save(data_dir + 'processed_data/mnist/X_train.npy', X_train)
        np.save(data_dir + 'processed_data/mnist/X_test.npy', X_test)
        np.save(data_dir + 'processed_data/mnist/Y_train.npy', Y_train)
        np.save(data_dir + 'processed_data/mnist/Y_test.npy', Y_test)

        logging.info('Preprocessed mnist data downloaded successfully and saved.')

        print('Preprocessed mnist data downloaded successfully and saved.')

    return (X_train, Y_train), (X_test, Y_test)


def load_cifar10_data():

    # Get current working directory
    cwd = os.getcwd()

    data_dir = cwd + '/data/'

    initiate_logger(data_dir + 'cifar10_load_log.txt')

    try:
        # Load preprocessed cifar10 data
        X_train = np.load(data_dir + 'processed_data/cifar10/X_train.npy')
        X_test = np.load(data_dir + 'processed_data/cifar10/X_test.npy')
        Y_train = np.load(data_dir + 'processed_data/cifar10/Y_train.npy')
        Y_test = np.load(data_dir + 'processed_data/cifar10/Y_test.npy')

        logging.info('Preprocessed cifar10 data loaded successfully')

        print('Preprocessed cifar10 data loaded successfully')

    except:

        logging.info('Preprocessed cifar10 data not found. Trying to download data...')

        print('Preprocessed cifar10 data not found. Trying to download data...')

        os.makedirs(data_dir + 'processed_data/cifar10/')

        # Load raw cifar10 data
        (X_train, Y_train), (X_test, Y_test) = cifar10.load_data()

        # Save the data
        np.save(data_dir + 'processed_data/cifar10/X_train.npy', X_train)
        np.save(data_dir + 'processed_data/cifar10/X_test.npy', X_test)
        np.save(data_dir + 'processed_data/cifar10/Y_train.npy', Y_train)
        np.save(data_dir + 'processed_data/cifar10/Y_test.npy', Y_test)

        logging.info('Preprocessed cifar10 data downloaded successfully and saved.')

        print('Preprocessed cifar10 data downloaded successfully and saved.')

    return (X_train, Y_train), (X_test, Y_test)


def load_cifar100_data():
    
    # Get current working directory
    cwd = os.getcwd()

    data_dir = cwd + '/data/'

    # data_dir = cwd.replace(os.getcwd().split('/')[-1],'data') + '/'

    initiate_logger(data_dir + 'cifar100_load_log.txt')

    try:
        # Load preprocessed cifar100 data
        X_train = np.load(data_dir + 'processed_data/cifar100/X_train.npy')
        X_test = np.load(data_dir + 'processed_data/cifar100/X_test.npy')
        Y_train = np.load(data_dir + 'processed_data/cifar100/Y_train.npy')
        Y_test = np.load(data_dir + 'processed_data/cifar100/Y_test.npy')

        logging.info('Preprocessed cifar100 data loaded successfully')

        print('Preprocessed cifar100 data loaded successfully')

    except:

        logging.info('Preprocessed cifar100 data not found. Trying to download data...')

        print('Preprocessed cifar100 data not found. Trying to download data...')

        os.makedirs(data_dir + 'processed_data/cifar100/')

        # Load raw cifar100 data
        (X_train, Y_train), (X_test, Y_test) = cifar100.load_data()

        # Save the data
        np.save(data_dir + 'processed_data/cifar100/X_train.npy', X_train)
        np.save(data_dir + 'processed_data/cifar100/X_test.npy', X_test)
        np.save(data_dir + 'processed_data/cifar100/Y_train.npy', Y_train)
        np.save(data_dir + 'processed_data/cifar100/Y_test.npy', Y_test)

        logging.info('Preprocessed cifar100 data downloaded successfully and saved.')

        print('Preprocessed cifar100 data downloaded successfully and saved.')

    return (X_train, Y_train), (X_test, Y_test)