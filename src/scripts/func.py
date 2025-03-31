import os
import librosa


def get_all_files(directory):
    """
    Рекурсивно получает список всех файлов в указанной директории.

    Аргументы:
        directory (str): Путь к директории.

    Возвращает:
        list: Список путей ко всем файлам в директории.
    """
    all_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            full_path = os.path.join(root, file)
            all_files.append(full_path)
    return all_files


def split_audio_to_windows(wav_data, window_size=1.0, step_size=0.25, sr=16000):
    """
    Разбивает аудиоданные на окна фиксированного размера с заданным шагом.

    Аргументы:
        wav_data (numpy.ndarray): Аудиоданные.
        window_size (float): Размер окна в секундах. По умолчанию 1.0.
        step_size (float): Шаг между окнами в секундах. По умолчанию 0.25.
        sr (int): Частота дискретизации аудиоданных. По умолчанию 16000.

    Возвращает:
        list: Список окон аудиоданных.
    """
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


def resample_audio(audio_data, orig_sr=48000, target_sr=16000):
    """
    Пересэмплирует аудиоданные до целевой частоты дискретизации.

    Аргументы:
        audio_data (numpy.ndarray): Аудиоданные.
        orig_sr (int): Исходная частота дискретизации. По умолчанию 48000.
        target_sr (int): Целевая частота дискретизации. По умолчанию 16000.

    Возвращает:
        numpy.ndarray: Пересэмплированные аудиоданные.
    """
    return librosa.resample(audio_data, orig_sr=orig_sr, target_sr=target_sr)