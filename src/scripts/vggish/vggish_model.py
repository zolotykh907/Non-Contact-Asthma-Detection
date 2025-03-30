import tensorflow as tf

class VGGish(tf.keras.Model):
    def __init__(self, embedding_size=128):
        super(VGGish, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')
        self.pool1 = tf.keras.layers.MaxPool2D((2, 2), strides=2, padding='same')

        self.conv2 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')
        self.pool2 = tf.keras.layers.MaxPool2D((2, 2), strides=2, padding='same')

        self.conv3_1 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same')
        self.conv3_2 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same')
        self.pool3 = tf.keras.layers.MaxPool2D((2, 2), strides=2, padding='same')

        self.conv4_1 = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same')
        self.conv4_2 = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same')
        self.pool4 = tf.keras.layers.MaxPool2D((2, 2), strides=2, padding='same')

        # self.flatten = tf.keras.layers.Flatten()
        # self.fc1 = tf.keras.layers.Dense(4096, activation='relu')
        # self.fc2 = tf.keras.layers.Dense(4096, activation='relu')
        # self.embedding = tf.keras.layers.Dense(embedding_size, activation=None, name='embedding')

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.pool2(x)

        x = self.conv3_1(x)
        x = self.conv3_2(x)
        x = self.pool3(x)

        x = self.conv4_1(x)
        x = self.conv4_2(x)
        x = self.pool4(x)

        # x = self.flatten(x)
        # x = self.fc1(x)
        # x = self.fc2(x)
        # return self.embedding(x)

        return x

def load_vggish_checkpoint(model, checkpoint_path):
    """
    Загружает чекпоинт для модели VGGish.

    Эта функция восстанавливает веса модели VGGish из файла чекпоинта.
    Она использует функциональность TensorFlow Checkpoint для загрузки
    сохраненных весов и применяет их к предоставленной модели.

    Параметры:
    model (tf.keras.Model): 
        Модель VGGish, к которой будет применен чекпоинт.
    checkpoint_path (str): 
        Путь к файлу чекпоинта.

    Возвращает:
    None

    Примечание:
    Функция выводит подтверждающее сообщение после успешной загрузки чекпоинта.
    """
    ckpt = tf.train.Checkpoint(model=model)
    ckpt.restore(checkpoint_path).expect_partial()
    print("Checkpoint restored from", checkpoint_path)

