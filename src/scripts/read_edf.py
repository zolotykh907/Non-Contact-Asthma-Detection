import mne
import warnings
import numpy as np
import pyedflib
from typing import Tuple

warnings.filterwarnings("ignore")


def get_channel(filepath: str, channel_id: str = 'Mic') -> Tuple[np.ndarray, mne.Info]:
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


def pyedflib_get_channels(file_path: str, channel_id: int = 17) -> np.ndarray:
    """
    Извлекает данные указанного канала из EDF-файла с использованием библиотеки pyedflib.

    Параметры:
    -----------
    file_path : str
        Путь к файлу в формате EDF.
    channel_id : int, optional
        Индекс канала, данные которого нужно извлечь. По умолчанию 17.

    Возвращает:
    -----------
    signal_data : numpy.ndarray
        Данные выбранного канала.
    """
    f = pyedflib.EdfReader(file_path)

    # Чтение данных указанного канала
    signal_data = f.readSignal(channel_id)

    # Закрытие файла
    f.close()

    return signal_data
