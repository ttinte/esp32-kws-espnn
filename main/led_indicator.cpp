#include "led_indicator.h"

#include <stdint.h>

#include "app_state.h"
#include "esp_log.h"
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
    return;
  }
  led_strip_clear(s_strip);
  ESP_LOGI(TAG, "RGB indicator ready on GPIO%d", app_state::RGB_LED_PIN);
}

void update() {
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

}  // namespace led_indicator
