#pragma once

namespace led_indicator {

// Khoi tao LED RGB on-board (WS2812 qua RMT), LCD 16x2 va LED 7 doan.
void init();

// Goi moi vong Core 0: cap nhat RGB wake + render LCD + 7 doan dem nguoc.
void update();

// Dieu khien LCD tu presence_service (ngu/danh thuc khi tiet kiem dien).
void sleepLcd();
void wakeLcd();

// voice_service goi de hien preview lenh tren LCD trong vai giay.
void queueCommand(const char *command);

}  // namespace led_indicator
