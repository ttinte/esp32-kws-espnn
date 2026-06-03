#pragma once

#include <stdint.h>

namespace lcd1602 {

// Driver LCD 16x2 HD44780 qua PCF8574 (I2C 4-bit). init() tra ve false
// neu khong tim thay PCF8574 o LCD_ADDRESS -> caller bo qua moi thao tac.
bool init();
bool present();
void clear();
void print(uint8_t row, const char *text);
void setBacklight(bool on);
void setDisplay(bool on);

}  // namespace lcd1602
