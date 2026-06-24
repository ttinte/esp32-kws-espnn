# esp32-kws-espnn

Offline keyword spotting and voice-controlled light/fan firmware for ESP32-S3. Audio from an INMP441 microphone is converted into log-Mel features and processed by a fully quantized int8 TensorFlow Lite Micro model accelerated by ESP-NN.

## Voice Commands

Say the wake word `zero`, then say one command within 7.5 seconds:

| Model label | Action | Confidence threshold |
| --- | --- | ---: |
| `bat` | Turn the light on | 0.78 |
| `tat` | Turn the light off | 0.78 |
| `quay` | Turn the fan on | 0.50 |
| `dung` | Turn the fan off | 0.50 |
| `zero` | Activate or refresh the command window | 0.25 |
| `other` | Ignore | — |

Wake detection requires at least two positive results within the latest three inference results. A control command is executed after the same command label is detected twice consecutively. Commands are ignored during the first 300 ms after wake activation.

## Model and Audio Pipeline

- Model format: fully quantized int8 TensorFlow Lite
- Input shape: `1 × 49 × 24 × 1`
- Output classes: `bat`, `dung`, `quay`, `tat`, `zero`, `other`
- Audio window: 1 second, 16,000 samples at 16 kHz
- Capture buffer: 2-second ring buffer
- Features: 49 log-Mel frames with 24 Mel bins per frame
- Frame length / step / FFT length: 512 / 320 / 512 samples
- Mel frequency range: 20–7,600 Hz
- Tensor arena: 48 KiB
- Embedded model: `main/model_tiny.cpp`

The microphone task runs on core 0, while the KWS task runs on core 1. Signal preprocessing uses ESP-DSP, and supported TensorFlow Lite Micro kernels use the ESP-NN backend.

## Requirements

- ESP-IDF 5.3.5
- ESP32-S3 toolchain installed through ESP-IDF
- ESP-IDF Python environment

The following components are declared in `main/idf_component.yml` and downloaded by the ESP-IDF Component Manager:

- `espressif/esp-tflite-micro`
- `espressif/esp-nn`
- `espressif/esp-dsp`
- `espressif/led_strip`

## Build, Flash, and Monitor

Initialize the ESP-IDF environment, then run:

```bash
# Required once for a fresh repository
idf.py set-target esp32s3

# Configure dependencies and build
idf.py reconfigure
idf.py build

# Replace the serial port when necessary
idf.py -p /dev/ttyUSB0 flash monitor
```

Press `Ctrl+]` to exit the serial monitor.

The settings in `sdkconfig.defaults` enable ESP-NN optimization, compiler performance optimization, and a 240 MHz CPU frequency.

## Project Structure

```text
include/
├── app_state.h            Shared pins, timing, audio configuration, and state
├── kws_engine.h           KWS engine API and inference results
├── kws_model_config.h     Labels, actions, and confidence thresholds
├── lcd1602.h              HD44780/PCF8574 LCD API
├── led_indicator.h        RGB LED and LCD presentation API
├── presence_service.h     PIR sensor and button service API
└── voice_service.h        Wake-word and command processing API

main/
├── main.cpp               GPIO initialization, task creation, and main loop
├── kws_engine.cpp         I2S capture, log-Mel extraction, and TFLM inference
├── voice_service.cpp      Wake state machine and light/fan command execution
├── presence_service.cpp   PIR handling, inactivity timeout, and load restoration
├── led_indicator.cpp      WS2812 animation and LCD status rendering
├── lcd1602.cpp            4-bit HD44780 driver through PCF8574
└── model_tiny.cpp         Embedded TFLite model data

kws_voice/scripts/         Dataset preparation, training, evaluation, and conversion
```

Generated datasets, trained models, evaluation results, build artifacts, managed components, and local `sdkconfig` files are excluded from Git.
