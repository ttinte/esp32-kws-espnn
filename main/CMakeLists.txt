idf_component_register(SRCS "main.cpp" "kws_engine.cpp" "voice_service.cpp" "model_tiny.cpp" "led_indicator.cpp" "presence_service.cpp" "lcd1602.cpp"
                    INCLUDE_DIRS "." "../include" "../managed_components/espressif__esp-tflite-micro"
                    REQUIRES esp-tflite-micro esp_driver_gpio esp_driver_i2s driver esp_timer led_strip esp_driver_i2c)
