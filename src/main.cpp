#include "app_state.h"
#include "driver/gpio.h"
#include "esp_log.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "kws_engine.h"
#include "voice_service.h"

namespace {
constexpr const char *kTag = "main";
}

namespace app_state {

bool wakeWordActive = false;
uint32_t wakeWordActivatedMs = 0;
uint32_t lastKwsIdleLogMs = 0;
const char *pendingWakeLabel = nullptr;
uint8_t pendingWakeFrames = 0;
const char *pendingCommandLabel = nullptr;
const char *pendingCommandAction = nullptr;
uint8_t pendingCommandFrames = 0;
bool wakeSettled = false;

uint32_t nowMs() {
  return pdTICKS_TO_MS(xTaskGetTickCount());
}

}  // namespace app_state

extern "C" void app_main(void) {
  ESP_LOGI(kTag, "Booting esp32-kws-espnn milestone 1");

  gpio_config_t outputConfig = {};
  outputConfig.pin_bit_mask = (1ULL << app_state::LIGHT_PIN) | (1ULL << app_state::FAN_PIN);
  outputConfig.mode = GPIO_MODE_OUTPUT;
  outputConfig.pull_up_en = GPIO_PULLUP_DISABLE;
  outputConfig.pull_down_en = GPIO_PULLDOWN_DISABLE;
  outputConfig.intr_type = GPIO_INTR_DISABLE;
  gpio_config(&outputConfig);

  gpio_set_level(static_cast<gpio_num_t>(app_state::LIGHT_PIN), 0);
  gpio_set_level(static_cast<gpio_num_t>(app_state::FAN_PIN), 0);

  gpio_config_t inputConfig = {};
  inputConfig.pin_bit_mask = (1ULL << app_state::PIR_PIN) | (1ULL << app_state::PIR_TOGGLE_BUTTON_PIN);
  inputConfig.mode = GPIO_MODE_INPUT;
  inputConfig.pull_up_en = GPIO_PULLUP_ENABLE;
  inputConfig.pull_down_en = GPIO_PULLDOWN_DISABLE;
  inputConfig.intr_type = GPIO_INTR_DISABLE;
  gpio_config(&inputConfig);

  const bool kwsReady = kws_engine::init();
  if (kwsReady) {
    kws_engine::setPaused(false);
  }
  ESP_LOGI(kTag, "KWS status: %s", kws_engine::status());

  while (true) {
    voice_service::update();
    vTaskDelay(pdMS_TO_TICKS(10));
  }
}
