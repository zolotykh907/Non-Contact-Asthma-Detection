# %%
import os
import re 
import sys
import mne
import matplotlib.pyplot as plt
import numpy as np
import librosa
import soundfile as sf
import tensorflow as tf
from tqdm import tqdm  



# %%
import requests

def download_file(url, save_path):
    response = requests.get(url)
    with open(save_path, 'wb') as file:
        file.write(response.content)

def get_file_size(url):
    response = requests.head(url)
    file_size = int(response.headers.get("Content-Length", 0))

    return file_size


rml_folder = '/var/data/apnea/rml/'
edf_folder = '/var/data/apnea/datasets/edf/'
url_file = '/var/data/apnea/778740145531650048.txt'

with open(url_file, "r") as f:
    url_lines = f.readlines()

rml_files = os.listdir(rml_folder)
sorted_rml_files = sorted(rml_files, key=lambda x: int(x.split('-')[0]))

good_edf_files = []
bad_edf_files = []
bad_rml_files = []

for rml_file in tqdm(sorted_rml_files):
    patient_name = rml_file.replace('.rml', '')

    for line in url_lines:
        if line.endswith('.edf\n'):
            if patient_name in line:
                url = line.strip()
                
                file_size = get_file_size(url)
                
                file_name = url.split('fileName=')[-1]
                save_path = os.path.join(edf_folder, file_name)

                if os.path.exists(save_path):
                    if os.path.getsize(save_path) == file_size:
                        #print(f'{file_name} файл загружен полностью')
                        good_edf_files.append(file_name) 
                    else:
                        bad_edf_files.append(file_name)
                        bad_rml_files.append(rml_file)
                        #print(f'{file_name} не скачался полностью')


selected_edf_files = []
cute_bad_rml_files = set([bad_edf_file.split('-')[0] for bad_edf_file in bad_edf_files])

for good_edf_file in good_edf_files:
    patient_name = good_edf_file.replace('.edf', '').split('-')[0]
    
    if patient_name not in cute_bad_rml_files:
        selected_edf_files.append(good_edf_file)

sorted_files = sorted(selected_edf_files, key=lambda x: (
    int(x.split('-')[0]),  # Числовая часть в начале имени
    int(x.split('%5B')[1].split('%5D')[0])  # Число второе
))

sorted_files

# %%
with open('selected_edf_files.txt', 'w') as file:
    for file_name in sorted_files:
        file.write(file_name + '\n')

# %%


# %%


# %% 


# %%



