from preprocess_helper import *
import zipfile


# Set parameters
sample_rate = 4800
number_seconds_to_extract = 10
seed = 42
number_iterations = 1
augmentation_probability = 1.0
augmentation_amount_noise = 2
augmentation_amount_gibbon = 10


data_source_directory = '/Users/Donovan/Documents/raw_data/'
base_directory = '/Users/Donovan/Documents/Masters/masters-ed02/clean_code/data/'

file_names = ['Test_Labels.zip', 'Test.zip', 'Train_Labels.zip', 'Train.zip']

for file_name in file_names:
    with zipfile.ZipFile(f'{data_source_directory}{file_name}') as zf:
        zf.extractall(f'{base_directory}raw_data/' + file_name[:file_name.find('.zip')].lower() + '/')

for dataset in ['train', 'test']:

    # Set paths for the dataset in question
    audio_directory = f'{base_directory}raw_data/{dataset}/'
    timestamp_directory = f'{base_directory}raw_data/{dataset}_labels/'
    save_location = f'{base_directory}pickled_data/{dataset}/'
    augment_directory = f'{base_directory}augmented_data/{dataset}/'
    augment_image_directory = f'{base_directory}augmented_image_data/{dataset}/'
    training_file = f'{base_directory}{dataset}ing_files.txt'

    # Excute preprocessing
    execute_preprocessing_all_files(training_file, audio_directory, 
                            sample_rate, timestamp_directory,
                            number_seconds_to_extract, save_location,
                            augmentation_amount_noise, augmentation_probability, 
                            augmentation_amount_gibbon, seed, augment_directory, augment_image_directory,
                            number_iterations)





