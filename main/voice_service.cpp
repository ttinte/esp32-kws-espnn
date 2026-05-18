#include "voice_service.h"

#include <string.h>

#include "app_state.h"
#include "driver/gpio.h"
#include "esp_log.h"
#include "kws_engine.h"

namespace voice_service {
namespace {
using namespace app_state;

const char *TAG = "voice_service";

void resetWakeConfirmation() {
  pendingWakeLabel = "";
  pendingWakeFrames = 0;
}

bool confirmWake(const kws_engine::Result &result) {
  if (!result.isWakeWord) {
    resetWakeConfirmation();
    return false;
  }

  if (strcmp(pendingWakeLabel, WAKE_WORD_LABEL) != 0) {
    pendingWakeLabel = WAKE_WORD_LABEL;
    pendingWakeFrames = 1;
    return WAKE_CONFIRMATION_FRAMES <= 1;
  }

  if (pendingWakeFrames < WAKE_CONFIRMATION_FRAMES) {
    ++pendingWakeFrames;
  }

  return pendingWakeFrames >= WAKE_CONFIRMATION_FRAMES;
}

void resetCommandConfirmation() {
  pendingCommandLabel = "";
  pendingCommandAction = nullptr;
  pendingCommandFrames = 0;
}

bool confirmCommand(const kws_engine::Result &result) {
  if (!result.hasCommand) {
    resetCommandConfirmation();
    return false;
  }

  if (strcmp(pendingCommandLabel, result.label) != 0) {
    pendingCommandLabel = result.label;
    pendingCommandAction = result.command;
    pendingCommandFrames = 1;
    return COMMAND_CONFIRMATION_FRAMES <= 1;
  }

  if (pendingCommandFrames < COMMAND_CONFIRMATION_FRAMES) {
    ++pendingCommandFrames;
  }

  pendingCommandAction = result.command;
  return pendingCommandFrames >= COMMAND_CONFIRMATION_FRAMES;
}

}  // namespace

const char *executeCommand(const char *command) {
  if (!command || command[0] == '\0') {
    return "ignored";
  }

  if (strcmp(command, "light_on") == 0) {
    gpio_set_level(static_cast<gpio_num_t>(LIGHT_PIN), 1);
    return "Light ON";
  }
  if (strcmp(command, "light_off") == 0) {
    gpio_set_level(static_cast<gpio_num_t>(LIGHT_PIN), 0);
    return "Light OFF";
  }
  if (strcmp(command, "fan_on") == 0) {
    gpio_set_level(static_cast<gpio_num_t>(FAN_PIN), 1);
    return "Fan ON";
  }
  if (strcmp(command, "fan_off") == 0) {
    gpio_set_level(static_cast<gpio_num_t>(FAN_PIN), 0);
    return "Fan OFF";
  }

  return "Unknown command";
}

void update() {
  if (wakeWordActive && nowMs() - wakeWordActivatedMs >= WAKE_LISTEN_WINDOW_MS) {
    wakeWordActive = false;
    wakeSettled = false;
    resetWakeConfirmation();
    resetCommandConfirmation();
    ESP_LOGI(TAG, "[Wake] Timeout, back to idle");
  }

  kws_engine::Result result{};
  if (!kws_engine::poll(result)) {
    return;
  }

  ESP_LOGI(TAG,
           "[KWS #%lu] label=%s conf=%.3f wake=%d cmd=%s top2=%s:%.3f top3=%s:%.3f",
           static_cast<unsigned long>(result.inferenceCount),
           result.label,
           result.confidence,
           result.isWakeWord,
           result.hasCommand ? result.command : "none",
           result.top2Label ? result.top2Label : "-",
           result.top2Score,
           result.top3Label ? result.top3Label : "-",
           result.top3Score);

  if (!wakeWordActive) {
    if (confirmWake(result)) {
      wakeWordActive = true;
      wakeSettled = false;
      wakeWordActivatedMs = nowMs();
      resetWakeConfirmation();
      resetCommandConfirmation();
      ESP_LOGI(TAG, "[Wake] Activated");
    }
    return;
  }

  if (nowMs() - wakeWordActivatedMs < WAKE_COOLDOWN_MS) {
    return;
  }

  if (result.isWakeWord) {
    wakeSettled = false;
    wakeWordActivatedMs = nowMs();
    resetCommandConfirmation();
    return;
  }

  if (!wakeSettled) {
    wakeSettled = true;
  }

  if (confirmCommand(result)) {
    wakeWordActive = false;
    wakeSettled = false;
    const char *command = pendingCommandAction ? pendingCommandAction : result.command;
    resetWakeConfirmation();
    resetCommandConfirmation();
    ESP_LOGI(TAG, "[Control] %s -> %s", command, executeCommand(command));
  }
}

}  // namespace voice_service
