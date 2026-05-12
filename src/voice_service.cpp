#include "voice_service.h"

#include <cstring>

#include "app_state.h"
#include "driver/gpio.h"
#include "esp_log.h"
#include "kws_engine.h"

namespace voice_service {
namespace {
using namespace app_state;

constexpr const char *kTag = "voice_service";

void resetWakeConfirmation() {
  pendingWakeLabel = nullptr;
  pendingWakeFrames = 0;
}

bool confirmWake(const kws_engine::Result &result) {
  if (!result.isWakeWord) {
    resetWakeConfirmation();
    return false;
  }

  if (pendingWakeLabel == nullptr || std::strcmp(pendingWakeLabel, WAKE_WORD_LABEL) != 0) {
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
  pendingCommandLabel = nullptr;
  pendingCommandAction = nullptr;
  pendingCommandFrames = 0;
}

bool confirmCommand(const kws_engine::Result &result) {
  if (!result.hasCommand || result.label == nullptr) {
    resetCommandConfirmation();
    return false;
  }

  if (pendingCommandLabel == nullptr || std::strcmp(pendingCommandLabel, result.label) != 0) {
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
  if (command == nullptr) {
    return "No command";
  }

  if (std::strcmp(command, "light_on") == 0) {
    gpio_set_level(static_cast<gpio_num_t>(LIGHT_PIN), 1);
    return "Light ON";
  }
  if (std::strcmp(command, "light_off") == 0) {
    gpio_set_level(static_cast<gpio_num_t>(LIGHT_PIN), 0);
    return "Light OFF";
  }
  if (std::strcmp(command, "fan_on") == 0) {
    gpio_set_level(static_cast<gpio_num_t>(FAN_PIN), 1);
    return "Fan ON";
  }
  if (std::strcmp(command, "fan_off") == 0) {
    gpio_set_level(static_cast<gpio_num_t>(FAN_PIN), 0);
    return "Fan OFF";
  }

  return "Unknown command";
}

void update() {
  const uint32_t now = nowMs();

  if (wakeWordActive && now - wakeWordActivatedMs >= WAKE_LISTEN_WINDOW_MS) {
    wakeWordActive = false;
    wakeSettled = false;
    resetWakeConfirmation();
    resetCommandConfirmation();
    ESP_LOGI(kTag, "Wake timeout, back to idle");
  }

  kws_engine::Result result{};
  if (!kws_engine::poll(result)) {
    if (now - lastKwsIdleLogMs >= 2000) {
      lastKwsIdleLogMs = now;
      ESP_LOGI(kTag, "KWS idle status=%s paused=%d wake=%d",
               kws_engine::status(),
               kws_engine::isPaused(),
               wakeWordActive);
    }
    return;
  }

  lastKwsIdleLogMs = now;

  const bool shouldLogEvent = result.hasCommand || result.isWakeWord || result.inferenceCount % 20 == 0;
  if (shouldLogEvent) {
    ESP_LOGI(kTag,
             "KWS event #%lu top1=%s/%.3f top2=%s/%.3f top3=%s/%.3f command=%s wake=%d audio(min=%d max=%d peak=%d rms=%.1f) latency(read=%lums mel=%lums invoke=%lums total=%lums)",
             static_cast<unsigned long>(result.inferenceCount),
             result.label ? result.label : "<null>",
             static_cast<double>(result.confidence),
             result.top2Label ? result.top2Label : "<null>",
             static_cast<double>(result.top2Score),
             result.top3Label ? result.top3Label : "<null>",
             static_cast<double>(result.top3Score),
             result.hasCommand && result.command ? result.command : "none",
             result.isWakeWord,
             result.audioMin,
             result.audioMax,
             result.audioPeak,
             static_cast<double>(result.audioRms),
             static_cast<unsigned long>(result.readMs),
             static_cast<unsigned long>(result.melMs),
             static_cast<unsigned long>(result.invokeMs),
             static_cast<unsigned long>(result.totalMs));
  }

  if (!wakeWordActive) {
    if (confirmWake(result)) {
      wakeWordActive = true;
      wakeSettled = false;
      wakeWordActivatedMs = now;
      resetWakeConfirmation();
      resetCommandConfirmation();
      ESP_LOGI(kTag, "Wake activated, listening for command");
    }
    return;
  }

  if (now - wakeWordActivatedMs < WAKE_COOLDOWN_MS) {
    return;
  }

  if (result.isWakeWord) {
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
    const char *commandResult = executeCommand(command);
    ESP_LOGI(kTag, "Control %s -> %s", command ? command : "<null>", commandResult);
  }
}

}  // namespace voice_service
