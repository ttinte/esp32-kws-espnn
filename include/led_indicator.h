#pragma once

namespace led_indicator {

// Khoi tao LED RGB on-board (WS2812 qua RMT).
void init();

// LCD power control
void sleepLcd();
void wakeLcd();

void update();

}  // namespace led_indicator