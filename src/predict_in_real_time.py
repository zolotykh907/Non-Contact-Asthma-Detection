import numpy as np
import librosa
import sounddevice as sd
from tensorflow.keras.models import load_model
import queue
import threading
import time

SAMPLE_RATE = 16000
DURATION = 5
SAMPLES_PER_SEGMENT = SAMPLE_RATE * DURATION
NUM_MFCC = 13
N_FFT = 2048
HOP_LENGTH = 512

audio_queue = queue.Queue()

model = load_model('/Users/igorzolotyh/Study/Non-Contact-Asthma-Detection/src/asthma_bi_lstm.h5')

def audio_callback(indata, frames, time, status):
    """Callback функция для записи аудио с микрофона"""
    if status:
        print(status, flush=True)
    audio_queue.put(indata.copy())

def extract_features(signal):
    """Извлечение MFCC признаков из аудиосигнала"""
    if len(signal) > SAMPLES_PER_SEGMENT:
        signal = signal[:SAMPLES_PER_SEGMENT]
    else:
        signal = np.pad(signal, (0, max(0, SAMPLES_PER_SEGMENT - len(signal))), 'constant')
    
    mfcc = librosa.feature.mfcc(y=signal, sr=SAMPLE_RATE, n_mfcc=NUM_MFCC, 
                                n_fft=N_FFT, hop_length=HOP_LENGTH)
    mfcc = mfcc.T 
    
    return np.expand_dims(mfcc, axis=0)

def predict_audio():
    """Основная функция для предсказания"""
    print("Starting audio recording... Press Ctrl+C to stop.")
    
    buffer = np.zeros((int(SAMPLE_RATE * (DURATION + 1)), 1), dtype=np.float32)
    
    with sd.InputStream(callback=audio_callback,
                       channels=1,
                       samplerate=SAMPLE_RATE,
                       blocksize=SAMPLE_RATE):
        
        while True:
            try:
                new_data = audio_queue.get()
                
                buffer[:-SAMPLE_RATE] = buffer[SAMPLE_RATE:]
                buffer[-SAMPLE_RATE:] = new_data
                
                segment = buffer[-SAMPLES_PER_SEGMENT:].flatten()
                
                features = extract_features(segment)
                
                prediction = model.predict(features)
                class_id = np.argmax(prediction)
                confidence = np.max(prediction)
                
                print(f"Predicted class: {class_id} (confidence: {confidence:.2f})")
                
                # Небольшая задержка для стабильности
                time.sleep(0.1)
                
            except KeyboardInterrupt:
                print("Stopping...")
                break
            except Exception as e:
                print(f"Error: {e}")
                continue

if __name__ == "__main__":
    print("Available audio devices:")
    print(sd.query_devices())
    
    predict_audio()