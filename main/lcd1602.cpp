#include "lcd1602.h"

#include "app_state.h"
#include "driver/i2c_master.h"
#include "esp_log.h"
#include "esp_rom_sys.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

namespace lcd1602 {
namespace {

constexpr char TAG[] = "lcd1602";

// Mapping bit cua PCF8574 backpack -> chan HD44780.
constexpr uint8_t LCD_RS = 0x01;  // P0
constexpr uint8_t LCD_EN = 0x04;  // P2
constexpr uint8_t LCD_BL = 0x08;  // P3 (backlight)

i2c_master_bus_handle_t s_bus = nullptr;
i2c_master_dev_handle_t s_dev = nullptr;
bool s_present = false;
uint8_t s_backlight = LCD_BL;

void pcfWrite(uint8_t data) {
  i2c_master_transmit(s_dev, &data, 1, 50);
}

// Chot 4 bit cao (value da chua data nibble + RS + backlight, chua co EN).
void pulse(uint8_t value) {
  pcfWrite(value | LCD_EN);
  esp_rom_delay_us(1);
  pcfWrite(value & ~LCD_EN);
  esp_rom_delay_us(50);
}

// Gui 1 byte lenh/du lieu o che do 4-bit: nibble cao roi nibble thap.
void send(uint8_t value, uint8_t mode) {
  pulse((value & 0xF0) | mode | s_backlight);
  pulse(((value << 4) & 0xF0) | mode | s_backlight);
}

}  // namespace

bool init() {
  i2c_master_bus_config_t busConfig = {};
  busConfig.i2c_port = I2C_NUM_0;
  busConfig.sda_io_num = static_cast<gpio_num_t>(app_state::I2C_SDA_PIN);
  busConfig.scl_io_num = static_cast<gpio_num_t>(app_state::I2C_SCL_PIN);
  busConfig.clk_source = I2C_CLK_SRC_DEFAULT;
  busConfig.glitch_ignore_cnt = 7;
  busConfig.flags.enable_internal_pullup = true;
  if (i2c_new_master_bus(&busConfig, &s_bus) != ESP_OK) {
    ESP_LOGE(TAG, "I2C bus init failed");
    return false;
  }

  if (i2c_master_probe(s_bus, app_state::LCD_ADDRESS, 100) != ESP_OK) {
    ESP_LOGW(TAG, "LCD 0x%02X not found", app_state::LCD_ADDRESS);
    return false;
  }

  i2c_device_config_t devConfig = {};
  devConfig.dev_addr_length = I2C_ADDR_BIT_LEN_7;
  devConfig.device_address = app_state::LCD_ADDRESS;
  devConfig.scl_speed_hz = 100000;
  if (i2c_master_bus_add_device(s_bus, &devConfig, &s_dev) != ESP_OK) {
    ESP_LOGE(TAG, "I2C add device failed");
    return false;
  }

  s_present = true;
  s_backlight = LCD_BL;

  // Init 4-bit theo datasheet HD44780.
  vTaskDelay(pdMS_TO_TICKS(50));
  pulse(0x30 | s_backlight);
  vTaskDelay(pdMS_TO_TICKS(5));
  pulse(0x30 | s_backlight);
  esp_rom_delay_us(150);
  pulse(0x30 | s_backlight);
  esp_rom_delay_us(150);
  pulse(0x20 | s_backlight);  // chuyen sang 4-bit

  send(0x28, 0);  // 4-bit, 2 dong, font 5x8
  send(0x08, 0);  // tat hien thi
  send(0x01, 0);  // clear
  vTaskDelay(pdMS_TO_TICKS(2));
  send(0x06, 0);  // entry: tang dia chi, khong dich
  send(0x0C, 0);  // bat hien thi, tat con tro

  ESP_LOGI(TAG, "LCD 0x%02X ready on SDA=%d SCL=%d",
           app_state::LCD_ADDRESS, app_state::I2C_SDA_PIN, app_state::I2C_SCL_PIN);
  return true;
}

bool present() { return s_present; }

void clear() {
  if (!s_present) return;
  send(0x01, 0);
  vTaskDelay(pdMS_TO_TICKS(2));
}

void print(uint8_t row, const char *text) {
  if (!s_present) return;
  send(row == 0 ? 0x80 : 0xC0, 0);
  for (const char *p = text; *p; ++p) {
    send(static_cast<uint8_t>(*p), LCD_RS);
  }
}

void setBacklight(bool on) {
  s_backlight = on ? LCD_BL : 0;
  if (s_present) pcfWrite(s_backlight);
}

void setDisplay(bool on) {
  if (!s_present) return;
  send(on ? 0x0C : 0x08, 0);
}

}  // namespace lcd1602
