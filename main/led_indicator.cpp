#include "led_indicator.h"

#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "app_state.h"
#include "driver/gpio.h"
#include "esp_log.h"
#include "lcd1602.h"
#include "led_strip.h"

namespace led_indicator {
namespace {

constexpr char TAG[] = "led_indicator";

constexpr uint32_t kHueStepMs = 20;
constexpr uint16_t kHueStep = 12;
constexpr uint8_t kBrightness = 48;

led_strip_handle_t s_strip = nullptr;
bool s_active = false;
uint16_t s_hue = 0;
uint32_t s_stepAtMs = 0;

// ── LCD presentation ──────────────────────────────────────────
char s_lastLine1[17] = {0};
char s_lastLine2[17] = {0};
bool s_lcdSleepLocked = false;
const char *s_pendingCommand = nullptr;
uint32_t s_pendingCommandAtMs = 0;

// ── 7-segment (74HC595) ───────────────────────────────────────
constexpr uint8_t kDigitMap[10] = {
    0x3F, 0x06, 0x5B, 0x4F, 0x66, 0x6D, 0x7D, 0x07, 0x7F, 0x6F};
int s_sevenSegLastValue = -2;

// Hue [0,1536) -> RGB
void hueToRgb(uint16_t hue, uint8_t &r, uint8_t &g, uint8_t &b) {
  const uint8_t section = (hue / 256) % 6;
  const uint8_t offset = hue % 256;
  const uint8_t down = 255 - offset;

  switch (section) {
    case 0: r = 255; g = offset; b = 0; break;
    case 1: r = down; g = 255; b = 0; break;
    case 2: r = 0; g = 255; b = offset; break;
    case 3: r = 0; g = down; b = 255; break;
    case 4: r = offset; g = 0; b = 255; break;
    default: r = 255; g = 0; b = down; break;
  }
}

void clearState() {
  s_active = false;
  s_hue = 0;
  s_stepAtMs = 0;
}

void updateRgb() {
  if (s_strip == nullptr) {
    return;
  }

  if (app_state::wakeWordActive) {
    if (!s_active) {
      s_active = true;
      s_hue = 0;
      s_stepAtMs = 0;
    }

    const uint32_t now = app_state::nowMs();
    if (s_stepAtMs != 0 && (now - s_stepAtMs) < kHueStepMs) {
      return;
    }
    s_stepAtMs = now;
    s_hue = static_cast<uint16_t>((s_hue + kHueStep) % 1536);

    uint8_t r = 0, g = 0, b = 0;
    hueToRgb(s_hue, r, g, b);
    r = static_cast<uint8_t>((static_cast<uint16_t>(r) * kBrightness) / 255);
    g = static_cast<uint8_t>((static_cast<uint16_t>(g) * kBrightness) / 255);
    b = static_cast<uint8_t>((static_cast<uint16_t>(b) * kBrightness) / 255);
    led_strip_set_pixel(s_strip, 0, r, g, b);
    led_strip_refresh(s_strip);
    return;
  }

  if (s_active) {
    clearState();
    led_strip_clear(s_strip);
  }
}

// ── LCD helpers ───────────────────────────────────────────────
const char *buildPirLine() {
  using namespace app_state;
  if (!pirEnabled) return "PIR:OFF";
  if (fullPowerOffActive) return "PIR:OFF FULL";
  if (tempOffActive) return "PIR:COUNTDOWN";
  if (pirMotionDetected) return "PIR:MOVE";
  return "PIR:IDLE";
}

const char *currentPreview() {
  if (s_pendingCommand == nullptr || s_pendingCommand[0] == '\0') return "";
  if (app_state::nowMs() - s_pendingCommandAtMs >= app_state::LCD_COMMAND_PREVIEW_MS) return "";
  if (strcmp(s_pendingCommand, "light_on") == 0) return "L_ON";
  if (strcmp(s_pendingCommand, "light_off") == 0) return "L_OFF";
  if (strcmp(s_pendingCommand, "fan_on") == 0) return "F_ON";
  if (strcmp(s_pendingCommand, "fan_off") == 0) return "F_OFF";
  return s_pendingCommand;
}

void renderStatus() {
  if (s_lcdSleepLocked || !lcd1602::present()) return;

  char line1[17];
  char line2[17];
  const char *top = app_state::wakeWordActive ? "Say command..." : app_state::LCD_DEFAULT_TOP_LINE;
  snprintf(line1, sizeof(line1), "%-16.16s", top);
  snprintf(line2, sizeof(line2), "%-8.8s %-7.7s", buildPirLine(), currentPreview());

  if (strcmp(line1, s_lastLine1) == 0 && strcmp(line2, s_lastLine2) == 0) return;

  lcd1602::print(0, line1);
  lcd1602::print(1, line2);
  strcpy(s_lastLine1, line1);
  strcpy(s_lastLine2, line2);
}

// ── 7-segment helpers ─────────────────────────────────────────
void initSevenSegGpio() {
  gpio_config_t io = {};
  io.mode = GPIO_MODE_OUTPUT;
  io.pin_bit_mask = (1ULL << app_state::SHIFT595_DATA_PIN) |
                    (1ULL << app_state::SHIFT595_CLOCK_PIN) |
                    (1ULL << app_state::SHIFT595_LATCH_PIN);
  gpio_config(&io);
  gpio_set_level(static_cast<gpio_num_t>(app_state::SHIFT595_LATCH_PIN), 1);
  gpio_set_level(static_cast<gpio_num_t>(app_state::SHIFT595_CLOCK_PIN), 0);
}

void shiftOutByte(uint8_t value) {  // MSB first
  for (int i = 7; i >= 0; --i) {
    gpio_set_level(static_cast<gpio_num_t>(app_state::SHIFT595_DATA_PIN), (value >> i) & 1);
    gpio_set_level(static_cast<gpio_num_t>(app_state::SHIFT595_CLOCK_PIN), 1);
    gpio_set_level(static_cast<gpio_num_t>(app_state::SHIFT595_CLOCK_PIN), 0);
  }
}

void writeSevenSegRaw(uint8_t tensRaw, uint8_t onesRaw) {
  gpio_set_level(static_cast<gpio_num_t>(app_state::SHIFT595_LATCH_PIN), 0);
  shiftOutByte(tensRaw);
  shiftOutByte(onesRaw);
  gpio_set_level(static_cast<gpio_num_t>(app_state::SHIFT595_LATCH_PIN), 1);
}

uint8_t encodeSevenSegDigit(uint8_t digit) {
  if (digit > 9) return 0x00;
  uint8_t value = kDigitMap[digit];
  if (app_state::SEVEN_SEG_COMMON_ANODE) value = static_cast<uint8_t>(~value);
  return value;
}

void showSevenSegNumber(int value) {
  if (value < 0) {
    if (s_sevenSegLastValue == -1) return;
    const uint8_t off = app_state::SEVEN_SEG_COMMON_ANODE ? 0xFF : 0x00;
    writeSevenSegRaw(off, off);
    s_sevenSegLastValue = -1;
    return;
  }
  if (value > 99) value = 99;

  if (value == 0) {
    if (s_sevenSegLastValue == -3) return;
    const uint8_t off = app_state::SEVEN_SEG_COMMON_ANODE ? 0xFF : 0x00;
    uint8_t ones = off;
    if (app_state::SEVEN_SEG_COMMON_ANODE) {
      ones = static_cast<uint8_t>(ones & ~0x80);
    } else {
      ones = static_cast<uint8_t>(ones | 0x80);
    }
    writeSevenSegRaw(off, ones);
    s_sevenSegLastValue = -3;
    return;
  }

  if (s_sevenSegLastValue == value) return;

  const uint8_t tens = static_cast<uint8_t>(value / 10);
  const uint8_t ones = static_cast<uint8_t>(value % 10);
  writeSevenSegRaw(encodeSevenSegDigit(tens), encodeSevenSegDigit(ones));
  s_sevenSegLastValue = value;
}

uint32_t remainingPresenceMs() {
  if (!app_state::tempOffActive) return app_state::NO_PRESENCE_TIMEOUT_MS;
  const uint32_t elapsed = app_state::nowMs() - app_state::tempOffStartedMs;
  if (elapsed >= app_state::FULL_OFF_DELAY_MS) return 0;
  return app_state::FULL_OFF_DELAY_MS - elapsed;
}

void updateSevenSeg() {
  using namespace app_state;
  if (!pirEnabled) {
    showSevenSegNumber(-1);
    return;
  }
  if (fullPowerOffActive) {
    showSevenSegNumber(0);
    return;
  }
  if (tempOffActive) {
    const uint32_t remaining = remainingPresenceMs();
    showSevenSegNumber(static_cast<int>((remaining + 999) / 1000));
    return;
  }
  showSevenSegNumber(-1);
}

}  // namespace

void init() {
  led_strip_config_t strip_config = {};
  strip_config.strip_gpio_num = app_state::RGB_LED_PIN;
  strip_config.max_leds = 1;

  led_strip_rmt_config_t rmt_config = {};
  rmt_config.resolution_hz = 10 * 1000 * 1000;  // 10MHz

  esp_err_t err = led_strip_new_rmt_device(&strip_config, &rmt_config, &s_strip);
  if (err != ESP_OK) {
    ESP_LOGE(TAG, "led_strip init failed: %s", esp_err_to_name(err));
    s_strip = nullptr;
  } else {
    led_strip_clear(s_strip);
    ESP_LOGI(TAG, "RGB indicator ready on GPIO%d", app_state::RGB_LED_PIN);
  }

  initSevenSegGpio();
  showSevenSegNumber(-1);

  lcd1602::init();
  renderStatus();
}

void update() {
  updateRgb();
  renderStatus();
  updateSevenSeg();
}

void sleepLcd() {
  s_lcdSleepLocked = true;
  lcd1602::clear();
  lcd1602::setDisplay(false);
  lcd1602::setBacklight(false);
  s_lastLine1[0] = '\0';
  s_lastLine2[0] = '\0';
}

void wakeLcd() {
  s_lcdSleepLocked = false;
  lcd1602::setDisplay(true);
  lcd1602::setBacklight(true);
  s_lastLine1[0] = '\0';
  s_lastLine2[0] = '\0';
  renderStatus();
}

void queueCommand(const char *command) {
  s_pendingCommand = command;
  s_pendingCommandAtMs = app_state::nowMs();
}

}  // namespace led_indicator
