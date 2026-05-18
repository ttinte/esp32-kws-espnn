#include <stdio.h>

#include "app_state.h"
#include "driver/gpio.h"
#include "esp_log.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "kws_engine.h"
#include "voice_service.h"

namespace app_state {
bool wakeWordActive = false;
uint32_t wakeWordActivatedMs = 0;
uint32_t lastKwsIdleLogMs = 0;
const char *pendingWakeLabel = "";
uint8_t pendingWakeFrames = 0;
const char *pendingCommandLabel = "";
const char *pendingCommandAction = nullptr;
uint8_t pendingCommandFrames = 0;
bool wakeSettled = false;

uint32_t nowMs() { return static_cast<uint32_t>(xTaskGetTickCount() * portTICK_PERIOD_MS); }
}  // namespace app_state

extern "C" void app_main(void) {
  gpio_config_t io_conf = {};
  io_conf.mode = GPIO_MODE_OUTPUT;
  io_conf.pin_bit_mask = (1ULL << app_state::LIGHT_PIN) | (1ULL << app_state::FAN_PIN);
  gpio_config(&io_conf);
  gpio_set_level(static_cast<gpio_num_t>(app_state::LIGHT_PIN), 0);
  gpio_set_level(static_cast<gpio_num_t>(app_state::FAN_PIN), 0);

  if (!kws_engine::init()) {
    ESP_LOGE("main", "KWS init failed: %s", kws_engine::status());
    return;
  }

  kws_engine::setPaused(false);

  while (true) {
    voice_service::update();
    vTaskDelay(pdMS_TO_TICKS(20));
  }
}
