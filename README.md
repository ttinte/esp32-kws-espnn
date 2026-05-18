# esp32-kws-espnn

ESP32-S3 Keyword Spotting (KWS) firmware using ESP-IDF and TensorFlow Lite Micro.


## Prerequisites
- ESP-IDF v5.3.5 installed and initialized
- Target board: ESP32-S3
- INMP441 I2S digital microphone

## Build & Flash
```bash
# Set target (first time only)
idf.py set-target esp32s3

# Build
idf.py build

# Flash and monitor
idf.py -p /dev/ttyUSB0 flash monitor
```

## Architecture
- main/main.cpp: Application entry point
- main/kws_engine.cpp: Audio capture, Mel feature extraction, TFLite inference
- main/voice_service.cpp: Wake-word state machine and command handling
- main/model_tiny.cpp: Embedded TFLite model binary
- include/: Public headers

## Model
- Quantized int8 TFLite Micro model
- Input: 49×10 Mel spectrogram
- Classes: zero, on, off, yes, no, other

## Notes
- Uses legacy ESP-IDF I2S driver APIs
- Training pipeline assets are in kws_voice/