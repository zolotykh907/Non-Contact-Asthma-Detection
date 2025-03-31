import os
import time
import librosa
import numpy as np
import tensorflow as tf
from tqdm import tqdm

import sys
sys.path.append(os.path.abspath("..")) 
sys.path.append("/var/data/vggish/")

import vggish_input, vggish_slim

checkpoint_path = "/var/data/src/vggish/vggish_model.ckpt"  
pca_params_path = "/var/data/src/vggish/vggish_pca_params.npz"


def split_audio_to_windows(audio_path, window_size=1.0, step_size=0.25):
    """
    Разбивает аудиофайл на окна фиксированного размера с заданным шагом.

    Аргументы:
        audio_path (str): Путь к аудиофайлу.
        window_size (float): Размер окна в секундах. По умолчанию 1.0.
        step_size (float): Шаг между окнами в секундах. По умолчанию 0.25.

    Возвращает:
        tuple: Список окон аудиоданных и частота дискретизации.
    """
    wav_data, sample_rate = librosa.load(audio_path, sr=None)

    if len(wav_data.shape) > 1:
        wav_data = np.mean(wav_data, axis=1)

    window_samples = int(window_size * sample_rate)
    step_samples = int(step_size * sample_rate)
    num_windows = (len(wav_data) - window_samples) // step_samples + 1

    windows = []
    for i in range(num_windows):
        start = i * step_samples
        end = start + window_samples
        window = wav_data[start:end]
        windows.append(window)
    
    return windows, sample_rate


def windows_to_spectrograms(files, out_dir, window_size=1.0, step_size=0.25):
    """
    Преобразует окна аудиоданных в спектрограммы и сохраняет их в файлы.

    Аргументы:
        files (list): Список путей к аудиофайлам.
        out_dir (str): Директория для сохранения спектрограмм.
        window_size (float): Размер окна в секундах. По умолчанию 1.0.
        step_size (float): Шаг между окнами в секундах. По умолчанию 0.25.
    """
    with tf.Graph().as_default():
        sess = tf.compat.v1.Session()
        vggish_slim.define_vggish_slim(training=False)
        vggish_slim.load_vggish_slim_checkpoint(sess, checkpoint_path)

        features_tensor = sess.graph.get_tensor_by_name("vggish/input_features:0")
        pool4_output = sess.graph.get_tensor_by_name("vggish/pool4/MaxPool:0")

        for file in tqdm(files, desc="Обработка файлов"):
            output_file = os.path.join(out_dir, f"0/{file}_combined.npy")

            if os.path.exists(output_file):
                continue

            windows, sr = split_audio_to_windows(file)
            spectograms = []

            for i, window in enumerate(windows):
                mel_spec = vggish_input.waveform_to_examples(window, sr)

                [pool4_output_val] = sess.run(
                    [pool4_output],
                    feed_dict={features_tensor: mel_spec}
                )
                pool4_output_val = np.reshape(pool4_output_val, [-1, 6 * 4 * 512])

                spectograms.append(pool4_output_val)

            spectograms_combined = np.stack(spectograms, axis=0)
            spectograms_combined = np.squeeze(spectograms_combined, axis=1)

            data_for_save = {'spectograms': spectograms_combined, 'label': 0}
            np.save(output_file, data_for_save)
        
        sess.close()