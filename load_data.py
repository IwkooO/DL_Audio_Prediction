import pickle
import os
from tqdm import tqdm
import numpy as np

directory = 'data/train'

def load_data():
    length_list = []
    valence_values=[]
    recordings = []

    for filename in tqdm(os.listdir(directory)):
        if filename.endswith('.pkl'):
            file_path = os.path.join(directory, filename)
            with open(file_path, 'rb') as file:
                data = pickle.load(file)
                if data['valence'] != 2.333 and len(data['audio_data']) < 91000:
                    length_list.append(len(data['audio_data']))
                    valence_values.append(data['valence'])
                    recordings.append(np.trim_zeros(data['audio_data']))

    valence_values = np.array(valence_values)
    return recordings, valence_values

def pad(data):
    max_length = max(91000 for array in data) # Find the maximum length of the arrays
    padded_arrays = np.array([np.pad(array, (0, max_length - len(array)), mode='constant') for array in data])

    return padded_arrays