#pragma once

#include <stdint.h>

#include "kws_model_config.h"

namespace app_state {

constexpr uint8_t LIGHT_PIN = 40;
constexpr uint8_t FAN_PIN = 47;
constexpr uint8_t PIR_PIN = 21;
constexpr uint8_t PIR_TOGGLE_BUTTON_PIN = 42;
constexpr uint8_t I2C_SDA_PIN = 8;
constexpr uint8_t I2C_SCL_PIN = 9;
constexpr uint8_t I2S_BCK_PIN = 4;
constexpr uint8_t I2S_WS_PIN = 5;
constexpr uint8_t I2S_SD_PIN = 6;
constexpr uint8_t RGB_LED_PIN = 48;

constexpr uint32_t AUDIO_SAMPLE_RATE = 16000;
constexpr size_t AUDIO_DURATION_SAMPLES = 16000;
constexpr size_t MEL_FRAME_LEN = 512;
constexpr size_t MEL_FRAME_STEP = 320;
constexpr size_t MEL_FFT_LEN = 512;
constexpr size_t MEL_BINS = 24;
constexpr size_t MEL_N_FRAMES = 49;
constexpr size_t KWS_NUM_CLASSES = kws_model_config::kClassCount;
constexpr size_t KWS_TENSOR_ARENA_BYTES = 48 * 1024;

constexpr const char *WAKE_WORD_LABEL = "wake_up";
constexpr uint8_t WAKE_CONFIRMATION_FRAMES = 1;
constexpr uint8_t COMMAND_CONFIRMATION_FRAMES = 2;
constexpr uint8_t WAKE_VOTE_WINDOW = 3;
constexpr uint8_t WAKE_VOTE_MIN = 1;
constexpr uint32_t WAKE_LISTEN_WINDOW_MS = 15000;
constexpr uint32_t WAKE_COOLDOWN_MS = 900;
// Cong nang luong: bo qua frame im lang/nhieu nen INMP441 de khong nhan nham
// thanh lenh. kMicSampleShift=14 giu tieng noi xa cao hon nen nhung tranh clip.
constexpr float KWS_MIN_RMS = 120.0f;

// AGC: chuan hoa moi cua so 1s ve muc RMS dich truoc khi tinh log-mel, de
// model khong dung bien do lam dac trung (nho->other). VAD-gated: chi khuech
// dai khi RMS >= floor (tranh thoi phong im lang thanh lenh).
constexpr float AGC_TARGET_RMS = 4000.0f;
constexpr float AGC_VAD_FLOOR_RMS = 120.0f;
constexpr float AGC_MAX_GAIN = 12.0f;

constexpr uint8_t LCD_ADDRESS = 0x27;
constexpr uint8_t LCD_COLUMNS = 16;
constexpr uint8_t LCD_ROWS = 2;
constexpr const char *LCD_DEFAULT_TOP_LINE = "Awaiting wake..";
constexpr uint32_t LCD_COMMAND_PREVIEW_MS = 3000;

constexpr uint32_t NO_PRESENCE_TIMEOUT_MS = 15000;
constexpr uint32_t FULL_OFF_DELAY_MS = 30000;
constexpr uint32_t BUTTON_DEBOUNCE_MS = 50;

extern bool wakeWordActive;
extern uint32_t wakeWordActivatedMs;
extern uint32_t lastKwsIdleLogMs;
extern const char *pendingWakeLabel;
extern uint8_t pendingWakeFrames;
extern const char *pendingCommandLabel;
extern const char *pendingCommandAction;
extern uint8_t pendingCommandFrames;
extern bool wakeSettled;

extern bool pirEnabled;
extern bool tempOffActive;
extern bool fullPowerOffActive;
extern bool pirMotionDetected;
extern uint32_t tempOffStartedMs;

uint32_t nowMs();

}  // namespace app_state
