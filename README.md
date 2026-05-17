# esp32-kws-espnn

ESP32-S3 KWS firmware baseline using ESP-IDF.

## Prerequisites
- ESP-IDF v5.3.2 installed and initialized in the current shell environment.
- Target board: ESP32-S3.

## Build
```bash
idf.py set-target esp32s3
idf.py build
```

## Flash
```bash
idf.py -p <PORT> flash
```

## Monitor
```bash
idf.py -p <PORT> monitor
```

## Notes
- This repository is standardized for ESP-IDF workflow.
- KWS training pipeline assets are kept under `kws_voice/`.
