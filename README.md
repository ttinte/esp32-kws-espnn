# esp32-kws-espnn

KWS firmware for ESP32-S3 using ESP-IDF via PlatformIO.

## Build
```bash
platformio run
```

## Upload
```bash
platformio run --target upload
```

## Monitor
```bash
platformio device monitor
```

## Note
This project currently builds TensorFlowLite_ESP32 from the local PlatformIO library path:

- `~/.platformio/lib/TensorFlowLite_ESP32/src`

If a fresh machine does not have that library yet, install/copy the same `TensorFlowLite_ESP32` snapshot into:

- `~/.platformio/lib/TensorFlowLite_ESP32`
