#include "voice_service.h"

#include <string.h>

#include "app_state.h"
#include "driver/gpio.h"
#include "esp_heap_caps.h"
#include "esp_log.h"
#include "esp_system.h"
#include "kws_engine.h"
#include "led_indicator.h"

namespace voice_service {

const char *executeCommand(const char *command);

namespace {
using namespace app_state;

const char *TAG = "voice_service";
constexpr uint32_t kKwsLogEveryN = 10;
constexpr uint32_t kPerfLogEveryN = 10;
constexpr uint32_t kCommandDecisionDelayMs = 250;
constexpr uint32_t kBatDecisionDelayMs = 80;
constexpr uint32_t kDeferredCommandExpireMs = 600;
constexpr uint32_t kPostWakeModeIgnoreMs = 800;
constexpr uint32_t kCommandHoldMs = 550;
constexpr float kBatFastConfidence = 0.75f;
constexpr float kEarlyBatConfidence = 0.58f;
constexpr float kWeakWakeMinScore = 0.012f;
constexpr float kWeakWakeFastScore = 0.070f;
constexpr float kWeakWakePeakThreshold = 0.035f;
constexpr float kWeakWakeSumThreshold = 0.055f;
constexpr uint32_t kWeakWakeHoldMs = 650;
constexpr uint8_t kWeakWakeMinFrames = 2;

const char *deferredCommandAction = nullptr;
uint32_t deferredCommandReadyMs = 0;
uint32_t deferredCommandExpiresMs = 0;
uint32_t lastWakeModeExitMs = 0;
uint32_t pendingCommandLastMs = 0;
float weakWakeScoreSum = 0.0f;
float weakWakePeakScore = 0.0f;
uint32_t weakWakeLastMs = 0;
uint8_t weakWakeFrames = 0;

void resetWakeConfirmation() {
  pendingWakeLabel = "";
  pendingWakeFrames = 0;
}

void resetWeakWake() {
  weakWakeScoreSum = 0.0f;
  weakWakePeakScore = 0.0f;
  weakWakeLastMs = 0;
  weakWakeFrames = 0;
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

bool confirmWeakWake(const kws_engine::Result &result) {
  const uint32_t now = nowMs();
  if (weakWakeFrames > 0 &&
      weakWakeLastMs != 0 &&
      now - weakWakeLastMs > kWeakWakeHoldMs) {
    resetWeakWake();
  }

  if (result.wakeScore >= kWeakWakeMinScore) {
    weakWakeScoreSum += result.wakeScore;
    if (result.wakeScore > weakWakePeakScore) {
      weakWakePeakScore = result.wakeScore;
    }
    weakWakeLastMs = now;
    if (weakWakeFrames < UINT8_MAX) {
      ++weakWakeFrames;
    }
  }

  if (result.wakeScore >= kWeakWakeFastScore) {
    ESP_LOGI(TAG,
             "[Wake] Weak-fast evidence score=%.3f frames=%u",
             result.wakeScore,
             static_cast<unsigned int>(weakWakeFrames));
    resetWakeConfirmation();
    resetWeakWake();
    return true;
  }

  if (weakWakeFrames >= kWeakWakeMinFrames &&
      weakWakePeakScore >= kWeakWakePeakThreshold &&
      weakWakeScoreSum >= kWeakWakeSumThreshold) {
    ESP_LOGI(TAG,
             "[Wake] Weak evidence sum=%.3f peak=%.3f frames=%u",
             weakWakeScoreSum,
             weakWakePeakScore,
             static_cast<unsigned int>(weakWakeFrames));
    resetWakeConfirmation();
    resetWeakWake();
    return true;
  }

  return false;
}

void resetCommandConfirmation() {
  pendingCommandLabel = "";
  pendingCommandAction = nullptr;
  pendingCommandFrames = 0;
  pendingCommandLastMs = 0;
}

void resetDeferredCommand() {
  deferredCommandAction = nullptr;
  deferredCommandReadyMs = 0;
  deferredCommandExpiresMs = 0;
}

void expireDeferredCommand(uint32_t now) {
  if (deferredCommandAction &&
      deferredCommandExpiresMs != 0 &&
      static_cast<int32_t>(now - deferredCommandExpiresMs) >= 0) {
    resetDeferredCommand();
  }
}

void leaveWakeMode() {
  wakeWordActive = false;
  wakeSettled = false;
  lastWakeModeExitMs = nowMs();
  resetWakeConfirmation();
  resetWeakWake();
  resetCommandConfirmation();
  resetDeferredCommand();
}

void enterWakeMode() {
  wakeWordActive = true;
  wakeSettled = false;
  wakeWordActivatedMs = nowMs();
  resetWakeConfirmation();
  resetWeakWake();
  resetCommandConfirmation();
  resetDeferredCommand();
}

bool confirmCommand(const kws_engine::Result &result) {
  if (!result.hasCommand) {
    if (pendingCommandFrames > 0 &&
        pendingCommandLastMs != 0 &&
        nowMs() - pendingCommandLastMs > kCommandHoldMs) {
      resetCommandConfirmation();
    }
    return false;
  }

  if (strcmp(pendingCommandLabel, result.commandLabel) != 0) {
    pendingCommandLabel = result.commandLabel;
    pendingCommandAction = result.command;
    pendingCommandFrames = 1;
    pendingCommandLastMs = nowMs();
    return COMMAND_CONFIRMATION_FRAMES <= 1;
  }

  if (pendingCommandFrames < COMMAND_CONFIRMATION_FRAMES) {
    ++pendingCommandFrames;
  }

  pendingCommandAction = result.command;
  pendingCommandLastMs = nowMs();
  return pendingCommandFrames >= COMMAND_CONFIRMATION_FRAMES;
}

bool isBatCommand(const kws_engine::Result &result) {
  return result.hasCommand && strcmp(result.commandLabel, "bat") == 0;
}

void deferCommand(const char *command, uint32_t delayMs) {
  deferredCommandAction = command;
  const uint32_t cooldownReadyMs = wakeWordActivatedMs + WAKE_COOLDOWN_MS;
  const uint32_t decisionReadyMs = nowMs() + delayMs;
  deferredCommandReadyMs = decisionReadyMs > cooldownReadyMs ? decisionReadyMs : cooldownReadyMs;
  deferredCommandExpiresMs = deferredCommandReadyMs + kDeferredCommandExpireMs;
}

void runConfirmedCommand(const char *command) {
  ESP_LOGI(TAG, "[Control] %s -> %s", command, executeCommand(command));
  leaveWakeMode();
  ESP_LOGI(TAG, "[Wake] Command done, back to wake-word mode");
}

bool handleCommandResult(const kws_engine::Result &result) {
  expireDeferredCommand(nowMs());

  if (deferredCommandAction &&
      static_cast<int32_t>(nowMs() - deferredCommandReadyMs) >= 0) {
    runConfirmedCommand(deferredCommandAction);
    return true;
  }

  // "bat" ngan va de bi model day thanh top2/top3. Neu no la command tot
  // nhat va du manh, uu tien chot nhanh hon cac lenh khac.
  if (isBatCommand(result) && result.commandScore >= kBatFastConfidence) {
    deferCommand(result.command, kBatDecisionDelayMs);
    return true;
  }

  if (deferredCommandAction) {
    return true;
  }

  if (confirmCommand(result)) {
    const char *command = pendingCommandAction ? pendingCommandAction : result.command;
    deferCommand(command, kCommandDecisionDelayMs);
    return true;
  }

  return false;
}

bool hasDeferredCommandReady() {
  expireDeferredCommand(nowMs());
  return deferredCommandAction &&
         static_cast<int32_t>(nowMs() - deferredCommandReadyMs) >= 0;
}

}  // namespace

const char *executeCommand(const char *command) {
  if (!command || command[0] == '\0') {
    return "ignored";
  }

  led_indicator::queueCommand(command);

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
    leaveWakeMode();
    ESP_LOGI(TAG, "[Wake] Timeout, back to idle");
  }

  kws_engine::Result result{};
  if (!kws_engine::poll(result)) {
    return;
  }

  if (result.inferenceCount % kKwsLogEveryN == 0 || result.isWakeWord || result.hasCommand) {
    ESP_LOGI(TAG,
             "[KWS #%lu] label=%s conf=%.3f wake=%d ws=%.3f cmd=%s:%s %.3f rms=%.0f peak=%d top2=%s:%.3f top3=%s:%.3f",
             static_cast<unsigned long>(result.inferenceCount),
             result.label,
             result.confidence,
             result.isWakeWord,
             result.wakeScore,
             result.hasCommand ? result.commandLabel : "-",
             result.hasCommand ? result.command : "none",
             result.commandScore,
             result.audioRms,
             static_cast<int>(result.audioPeak),
             result.top2Label ? result.top2Label : "-",
             result.top2Score,
             result.top3Label ? result.top3Label : "-",
             result.top3Score);
  }

  if (result.inferenceCount % kPerfLogEveryN == 0) {
    ESP_LOGI(TAG,
             "[Perf #%lu][%s] read=%lums mel=%lums invoke=%lums total=%lums peak=%d rms=%.1f heap=%u min_heap=%u",
             static_cast<unsigned long>(result.inferenceCount),
             kws_engine::backendMode(),
             static_cast<unsigned long>(result.readMs),
             static_cast<unsigned long>(result.melMs),
             static_cast<unsigned long>(result.invokeMs),
             static_cast<unsigned long>(result.totalMs),
             static_cast<int>(result.audioPeak),
             result.audioRms,
             static_cast<unsigned int>(esp_get_free_heap_size()),
             static_cast<unsigned int>(esp_get_minimum_free_heap_size()));
  }

  // Cong nang luong: im lang/nhieu nen (rms thap) bi model gan nham thanh
  // quay/dung 0.9+. Bo qua nhung frame nay de khong wake/lenh oan.
  if (result.audioRms < KWS_MIN_RMS) {
    expireDeferredCommand(nowMs());
    if (weakWakeFrames > 0 &&
        weakWakeLastMs != 0 &&
        nowMs() - weakWakeLastMs > kWeakWakeHoldMs) {
      resetWeakWake();
    }
    return;
  }

  if (!wakeWordActive) {
    if (lastWakeModeExitMs != 0 && nowMs() - lastWakeModeExitMs < kPostWakeModeIgnoreMs) {
      resetWakeConfirmation();
      resetWeakWake();
      return;
    }
    if (confirmWake(result) || confirmWeakWake(result)) {
      enterWakeMode();
      ESP_LOGI(TAG, "[Wake] Activated, listening for command");
    }
    return;
  }

  // Sau khi wake_up vua kich hoat, bo qua duoi am cua chinh wake word.
  // Log thuc te cho thay duoi wake hay bi model gan thanh "tat", lam firmware
  // chay command ngay va thoat command mode.
  if (nowMs() - wakeWordActivatedMs < WAKE_COOLDOWN_MS) {
    resetCommandConfirmation();
    if (isBatCommand(result) && result.commandScore >= kEarlyBatConfidence) {
      deferCommand(result.command, kBatDecisionDelayMs);
    }
    return;
  }

  if (!wakeSettled) {
    wakeSettled = true;
    ESP_LOGI(TAG, "[Wake] Ready for command");
  }

  // Neu nguoi dung noi lai wake_up khi dang o command mode, xem nhu bat dau
  // lai cua so command moi thay vi im lang bo qua.
  if (result.isWakeWord) {
    if (hasDeferredCommandReady()) {
      handleCommandResult(result);
      return;
    }
    enterWakeMode();
    ESP_LOGI(TAG, "[Wake] Refreshed, listening for command");
    return;
  }

  handleCommandResult(result);
}

}  // namespace voice_service
