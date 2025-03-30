import xml.etree.ElementTree as ET
from typing import List, Dict


def get_attributes(
    file_path: str,
    space: str = 'ScoringData',
    namespace: str = 'http://www.respironics.com/PatientStudy.xsd',
    tag: str = "Events",
    family: str = 'Respiratory',
    types: List[str] = ['ObstructiveApnea', 'CentralApnea', 'MixedApnea', 'Hypopnea']
) -> List[Dict[str, str]]:
    """
    Извлекает атрибуты событий из XML-файла, соответствующих заданным критериям.

    Параметры:
    -----------
    file_path : str
        Путь к XML-файлу.
    space : str, optional
        Пространство в XML, где нужно искать атрибуты. По умолчанию 'ScoringData'.
    namespace : str, optional
        Пространство имен XML. По умолчанию 'http://www.respironics.com/PatientStudy.xsd'.
    tag : str, optional
        Тег, в котором находятся события. По умолчанию "Events".
    family : str, optional
        Семейство событий, которые нужно искать. По умолчанию 'Respiratory'.
    types : list, optional
        Список типов событий, которые нужно извлечь. По умолчанию ['ObstructiveApnea', 'CentralApnea', 'MixedApnea'].

    Возвращает:
    -----------
    list
        Список словарей с атрибутами событий, соответствующих заданным критериям.

    Исключения:
    -----------
    FileNotFoundError
        Если файл по указанному пути не найден.
    ET.ParseError
        Если файл не является валидным XML.
    """
    # Парсинг XML-файла
    tree = ET.parse(file_path)
    root = tree.getroot()

    # Поиск пространства с данными
    data = root.find(f'ns:{space}', {'ns': namespace})

    if data is None:
        return []

    target_arr = []

    # Поиск тега с событиями
    events = None
    for child in data:
        if child.tag.endswith(tag):
            events = child
            break

    if events is None:
        return []

    # Фильтрация событий по семейству и типу
    for event in events:
        event_attributes = event.attrib
        if event_attributes['Family'] == family and event_attributes['Type'] in types:
            target_arr.append(event_attributes)

    return target_arr