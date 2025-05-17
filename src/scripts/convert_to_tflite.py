import tensorflow as tf

model = tf.keras.models.load_model('/Users/igorzolotyh/Study/Non-Contact-Asthma-Detection/src/asthma_bi_lstm.h5')

converter = tf.lite.TFLiteConverter.from_keras_model(model)

converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
converter._experimental_lower_tensor_list_ops = False

tflite_model = converter.convert()

with open('asthma_model.tflite', 'wb') as f:
    f.write(tflite_model)

print("Модель успешно конвертирована в TFLite")