# # %%
# import os
# import re 
# import sys
# import mne
# import matplotlib.pyplot as plt
# import numpy as np
# import librosa
# import soundfile as sf
# import tensorflow as tf
# from tqdm import tqdm  

# from scripts.read_edf import get_channel
# from scripts.func import resample_audio

# # %%
# OUT_DIR = '/var/data/apnea/edf_channels/pulse/'
# EDF_DIR = '/var/data/apnea/datasets/edf/'

# txt_path = '/var/data/apnea/SafeVision-Apnea/selected_edf_files.txt'

# new_sr = 16000

# # %%
# with open(txt_path, "r") as f:
#     edf_files = f.readlines()

# edf_files = [edf_file.strip() for edf_file in edf_files]

# # %%
# os.makedirs(OUT_DIR, exist_ok=True)

# for edf_file in tqdm(edf_files):
#     patient_name = edf_file.split('%')[0] + '/'
#     output_path = (OUT_DIR + patient_name + edf_file).replace('.edf', '.wav')

#     if os.path.exists(output_path):
#         print(f'{output_path} уже существует, пропускаем')
#         continue
    
#     print(f'обрабатывается {output_path}')
#     os.makedirs(OUT_DIR+patient_name, exist_ok=True)
    
#     mic_channel, info = get_channel(EDF_DIR + edf_file, channel_id='Mic')
#     sr = info['sfreq']

#     mic_channel = resample_audio(mic_channel, orig_sr=sr, target_sr=new_sr)

#     sf.write(output_path, mic_channel, samplerate=new_sr)
    
#     del mic_channel


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

from scripts.read_edf import get_channel, pyedflib_get_channels
from scripts.func import resample_audio

# %%
OUT_DIR = '/var/data/apnea/edf_channels/pulse/'
EDF_DIR = '/var/data/apnea/datasets/edf/'

txt_path = '/var/data/apnea/SafeVision-Apnea/selected_edf_files.txt'

# %%
with open(txt_path, "r") as f:
    edf_files = f.readlines()

edf_files = [edf_file.strip() for edf_file in edf_files]

# %%
os.makedirs(OUT_DIR, exist_ok=True)

for edf_file in tqdm(edf_files):
    patient_name = edf_file.split('%')[0] + '/'
    output_path = (OUT_DIR + patient_name + edf_file).replace('.edf', '.npy')

    if os.path.exists(output_path):
        print(f'{output_path} уже существует, пропускаем')
        continue
    
    print(f'обрабатывается {output_path}')
    os.makedirs(OUT_DIR+patient_name, exist_ok=True)
    
    pulse_channel = pyedflib_get_channels(EDF_DIR + edf_file)

    np.save(output_path, pulse_channel)
    
    del pulse_channel