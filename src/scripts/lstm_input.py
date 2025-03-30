import mne
import os
import time
import warnings
import librosa
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf

from tqdm import tqdm   

from .func import split_audio_to_windows, get_all_files
from .vggish.vggish_input import waveform_to_examples
from .vggish.vggish_model import load_vggish_checkpoint, VGGish  


'''
Пример использования
--------------------

vggish = VGGish()
load_vggish_checkpoint(vggish, '/var/data/apnea/src/vggish/vggish_model.ckpt')

res = get_vggish_output('/var/data/apnea/datasets/new_mic_dataset/1/00001000-100507_0.wav', vggish)

res.shape = (17, 12288)
'''


def load_and_split_audio(wav, sr=16000):
    #audio, _ = librosa.load(audio_path, sr=sr)
    
    if len(wav) == 5 * sr:
        windows = split_audio_to_windows(wav)

        return windows

def windows_to_mel_spec(windows, sr=16000):
    if len(windows) == 17:
        mel_spectrograms = []

        for window in windows:
            mel_spectrogram = waveform_to_examples(window, sr)
            mel_spectrograms.append(mel_spectrogram)

        return np.array(mel_spectrograms)

def get_mel_spectrograms(wav, sr=16000):
    windows = load_and_split_audio(wav, sr)
    mel_spectrograms = windows_to_mel_spec(windows)
    mel_spectrograms = np.transpose(mel_spectrograms, (0, 2, 3, 1))

    return mel_spectrograms
    
def get_vggish_output(wav, vggish):
    mel_spectrograms = get_mel_spectrograms(wav)

    vggish_output = vggish(mel_spectrograms)
    vggish_output = np.reshape(vggish_output, [-1, 6 * 4 * 512])

    return vggish_output