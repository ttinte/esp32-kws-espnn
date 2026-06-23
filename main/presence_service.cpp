#include "presence_service.h"

#include "app_state.h"
#include "driver/gpio.h"
#include "kws_engine.h"
#include "led_indicator.h"

namespace presence_service {
namespace {
using namespace app_state;

uint32_t s_lastPresenceMs = 0;
int s_pirButtonStableState = 1;
int s_lastPirButtonReading = 1;
uint32_t s_lastPirButtonChangeMs = 0;
bool s_savedLightState = false;
bool s_savedFanState = false;

inline gpio_num_t pin(uint8_t p) { return static_cast<gpio_num_t>(p); }

void turnOffLoadsTemporarily() {
  if (!tempOffActive) {
    s_savedLightState = gpio_get_level(pin(LIGHT_PIN)) == 1;
    s_savedFanState = gpio_get_level(pin(FAN_PIN)) == 1;
  }
  led_indicator::sleepLcd();
  tempOffStartedMs = nowMs();
  tempOffActive = true;
  kws_engine::setPaused(true);
}

void turnOffLoadsFully() {
  gpio_set_level(pin(LIGHT_PIN), 0);
  gpio_set_level(pin(FAN_PIN), 0);
  tempOffActive = false;
  tempOffStartedMs = 0;
  led_indicator::sleepLcd();
  fullPowerOffActive = true;
}

void restoreLoads() {
  gpio_set_level(pin(LIGHT_PIN), s_savedLightState ? 1 : 0);
  gpio_set_level(pin(FAN_PIN), s_savedFanState ? 1 : 0);
  led_indicator::wakeLcd();
  tempOffActive = false;
  tempOffStartedMs = 0;
  fullPowerOffActive = false;
  kws_engine::setPaused(false);
}

void setPirEnabled(bool enabled) {
  pirEnabled = enabled;
  pirMotionDetected = false;
  s_lastPresenceMs = nowMs();
  tempOffStartedMs = 0;
  tempOffActive = false;

  if (!pirEnabled) {
    if (fullPowerOffActive) {
      fullPowerOffActive = false;
      led_indicator::wakeLcd();
    }
    kws_engine::setPaused(false);
  } else {
    kws_engine::setPaused(true);
  }
}

void toggleFromButton() {
  setPirEnabled(!pirEnabled);
  led_indicator::wakeLcd();
}

}  // namespace

void init() {
  gpio_config_t pirConf = {};
  pirConf.mode = GPIO_MODE_INPUT;
  pirConf.pin_bit_mask = (1ULL << PIR_PIN);
  gpio_config(&pirConf);

  gpio_config_t btnConf = {};
  btnConf.mode = GPIO_MODE_INPUT;
  btnConf.pin_bit_mask = (1ULL << PIR_TOGGLE_BUTTON_PIN);
  btnConf.pull_up_en = GPIO_PULLUP_ENABLE;
  gpio_config(&btnConf);

  s_pirButtonStableState = gpio_get_level(pin(PIR_TOGGLE_BUTTON_PIN));
  s_lastPirButtonReading = s_pirButtonStableState;
  s_lastPirButtonChangeMs = nowMs();
  s_lastPresenceMs = nowMs();
}

void pollButton() {
  const int reading = gpio_get_level(pin(PIR_TOGGLE_BUTTON_PIN));

  if (reading != s_lastPirButtonReading) {
    s_lastPirButtonChangeMs = nowMs();
    s_lastPirButtonReading = reading;
  }

  if (nowMs() - s_lastPirButtonChangeMs < BUTTON_DEBOUNCE_MS) {
    return;
  }

  if (reading != s_pirButtonStableState) {
    s_pirButtonStableState = reading;
    if (s_pirButtonStableState == 0) {  // LOW = nhan (pull-up)
      toggleFromButton();
    }
  }
}

void update() {
  if (!pirEnabled) {
    kws_engine::setPaused(false);
    return;
  }

  // Trong cua so nghe lenh, dong bang bo dem hien dien.
  if (wakeWordActive) {
    s_lastPresenceMs = nowMs();
    if (tempOffActive) {
      tempOffActive = false;
      tempOffStartedMs = 0;
      led_indicator::wakeLcd();
    }
  }

  const bool motionDetected = gpio_get_level(pin(PIR_PIN)) == 1;
  pirMotionDetected = motionDetected;

  if (motionDetected) {
    s_lastPresenceMs = nowMs();
    if (tempOffActive || fullPowerOffActive) {
      restoreLoads();
    } else {
      kws_engine::setPaused(false);
    }
  } else if (!wakeWordActive) {
    if (!tempOffActive && !fullPowerOffActive &&
        nowMs() - s_lastPresenceMs >= NO_PRESENCE_TIMEOUT_MS) {
      turnOffLoadsTemporarily();
    }

    if (tempOffActive &&
        nowMs() - tempOffStartedMs >= FULL_OFF_DELAY_MS) {
      turnOffLoadsFully();
    }
  }
}

}  // namespace presence_service
