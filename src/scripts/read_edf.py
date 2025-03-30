import mne
import warnings
import numpy as np
import pyedflib
from typing import Tuple

warnings.filterwarnings("ignore")


def get_channel(filepath: str, channel_id: str = 'Mic') -> Tuple[np.ndarray, mne.Info]: #PulseRate
    """
    Извлекает данные указанного канала из EDF-файла.

    Параметры:
    -----------
    filepath : str
        Путь к файлу в формате EDF.
    channel_id : str, optional
        Название канала, данные которого нужно извлечь. По умолчанию 'Mic'.

    Возвращает:
    -----------
    selected_data : numpy.ndarray
        Данные выбранного канала.
    info : mne.Info
        Информация о записи (метаданные).

    Исключения:
    -----------
    ValueError
        Если указанный канал отсутствует в файле.
    """
    # Загрузка данных из EDF-файла
    data = mne.io.read_raw_edf(filepath)
    
    # Проверка наличия канала
    if channel_id not in data.ch_names:
        raise ValueError(f"Канал '{channel_id}' отсутствует в файле.")
    
    # Извлечение данных указанного канала
    selected_channel = data.copy().pick_channels([channel_id])
    selected_data = selected_channel.get_data()[0]
    
    return selected_data, data.info

import pyedflib

def pyedflib_get_channels(file_path, channel_id=17): #17 - PulseRate
    f = pyedflib.EdfReader(file_path)

    #n = f.signals_in_file  # Количество сигналов (каналов)
    #signal_labels = f.getSignalLabels()  # Названия каналов
    #sfreqs = [f.getSampleFrequency(i) for i in range(n)]  # Частоты дискретизации

    signal_data = f.readSignal(channel_id)

    f.close()

    return signal_data
