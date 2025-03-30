import mne
import os
import time
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from tqdm import tqdm   
import xml.etree.ElementTree as ET
from scipy.io import wavfile
from scipy.signal import find_peaks
import warnings

import tensorflow as tf

from data_processing.read_rml import get_attributes

import sys
sys.path.append("/var/data/apnea/src/vggish")

# %%
import vggish_input, vggish_params, vggish_slim
from vggish.vggish_slim import define_vggish_slim, load_vggish_slim_checkpoint

checkpoint_path = "/var/data/apnea/src/vggish/vggish_model.ckpt"  
pca_params_path = "/var/data/apnea/src/vggish/vggish_pca_params.npz"

def split_audio_to_windows(wav_data, window_size=1.0, step_size=0.25, sr = 16000):
    window_samples = int(window_size * sr)
    step_samples = int(step_size * sr)
    num_windows = (len(wav_data) - window_samples) // step_samples + 1

    windows = []
    for i in range(num_windows):
        start = i * step_samples
        end = start + window_samples
        window = wav_data[start:end]
        windows.append(window)
    
    return windows


def get_spectrograms(wave, sr, reshape_flag=True):
    with tf.Graph().as_default():
        sess = tf.compat.v1.Session()
        vggish_slim.define_vggish_slim(training=False)
        vggish_slim.load_vggish_slim_checkpoint(sess, checkpoint_path)

        features_tensor = sess.graph.get_tensor_by_name("vggish/input_features:0")
        pool4_output = sess.graph.get_tensor_by_name("vggish/pool4/MaxPool:0")

        windows = split_audio_to_windows(wave)
        spectograms = []

        for i, window in enumerate(windows):
            mel_spec = vggish_input.waveform_to_examples(window, sr)

            [pool4_output_val] = sess.run([pool4_output],
                                        feed_dict={features_tensor: mel_spec})
            pool4_output_val = np.reshape(pool4_output_val, [-1, 6 * 4 * 512])

            spectograms.append(pool4_output_val)

        spectograms_combined = np.stack(spectograms, axis=0)
        spectograms_combined = np.squeeze(spectograms_combined, axis=1)

        sess.close()

    return spectograms_combined
