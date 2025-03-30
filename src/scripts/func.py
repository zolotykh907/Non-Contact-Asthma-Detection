import os
import librosa

def get_all_files(directory):
    """
    Извлекает пути для всех файлов в каталоге, включая файлы из подкаталогов.

    Параметры:
    -----------
    directory : str
        Путь к каталогу.

    Возвращает:
    -----------
    all_files: list
        Список путей к файлам.
    """
    all_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            full_path = os.path.join(root, file)
            all_files.append(full_path)
    return all_files


def split_audio_to_windows(wav_data, window_size=1.0, step_size=0.25, sr=16000):
    """
    Разбивает массив из WAV файла на окна определенной длины с определенным шагом

    Параметры:
    ----------
    wav_data : numpy.ndarray
        Массив нумпай из WAV файла.
    window_size : float
        размер окна.
    step_size : float
        размер шага.
    sr : int
        частота дискретизации WAV файла

    Возвращает:
    -----------
    windows: list
        Список окон.
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
    Пересэмплирует аудиоданные до заданной частоты.

    Аргументы:
    audio_data - массив сэмплов
    orig_sr    - исходная частота дискретизации
    target_sr  - целевая частота дискретизации

    Возвращает:
    - Пересэмплированные данные
    """
    return librosa.resample(audio_data, orig_sr=orig_sr, target_sr=target_sr)
