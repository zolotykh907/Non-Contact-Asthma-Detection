import tensorflow as tf
tf.get_logger().setLevel('ERROR')

from tensorflow.keras import layers, models

def create_bi_lstm_model(input_shape=(17, 12288)):
    """
    Создает Bi-LSTM модель для обработки временных последовательностей.

    Параметры:
    input_shape (tuple): Размерность входных данных (количество временных окон, размер признакового вектора).

    Возвращает:
    model (tensorflow.keras.Model): Скомпилированная модель Bi-LSTM.
    """
    model = models.Sequential()

    # Входной слой
    model.add(layers.Input(shape=input_shape))

    # TimeDistributed применяется к каждому временному шагу отдельно
    model.add(layers.TimeDistributed(layers.Dense(256, activation='relu')))  # Первый полносвязный слой
    model.add(layers.TimeDistributed(layers.Dense(128, activation='relu')))  # Второй полносвязный слой

    # Двунаправленный LSTM для обработки временной зависимости в обоих направлениях
    model.add(layers.Bidirectional(layers.LSTM(15, return_sequences=False)))

    # Dropout для предотвращения переобучения
    model.add(layers.Dropout(0.2))

    # Полносвязный слой с 64 нейронами и функцией активации ReLU
    model.add(layers.Dense(64, activation='relu'))

    # Выходной слой с 1 нейроном и сигмоидой для бинарной классификации
    model.add(layers.Dense(1, activation='sigmoid'))

    return model