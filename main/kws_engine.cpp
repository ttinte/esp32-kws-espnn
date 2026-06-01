#include "kws_engine.h"

#include <math.h>
#include <string.h>

#include "app_state.h"
#include "driver/i2s.h"
#include "dsps_fft2r.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/semphr.h"
#include "freertos/task.h"
#include "kws_model_config.h"
#include "model_tiny.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace kws_engine {
namespace {
using namespace app_state;

const char *TAG = "kws_engine";
constexpr float kPi = 3.14159265358979323846f;
constexpr size_t kFftBins = MEL_FFT_LEN / 2 + 1;
constexpr float kMelLowerHz = 20.0f;
constexpr float kMelUpperHz = 7600.0f;

#if CONFIG_NN_OPTIMIZED
constexpr const char *kBackendMode = "espnn";
#else
constexpr const char *kBackendMode = "baseline";
#endif

int16_t audioBuffer[AUDIO_DURATION_SAMPLES];
float hannWindow[MEL_FRAME_LEN];
float melWeights[MEL_BINS * kFftBins];
float fftPower[kFftBins];
__attribute__((aligned(16))) float fftCplx[MEL_FFT_LEN * 2];
float fftTwiddleRe[MEL_FFT_LEN / 2];
float fftTwiddleIm[MEL_FFT_LEN / 2];
float featureBuffer[MEL_N_FRAMES * MEL_BINS];
int8_t quantizedFeature[MEL_N_FRAMES * MEL_BINS];
alignas(16) uint8_t tensorArena[KWS_TENSOR_ARENA_BYTES];

const tflite::Model *model = nullptr;
tflite::MicroInterpreter *interpreter = nullptr;
TfLiteTensor *inputTensor = nullptr;
TfLiteTensor *outputTensor = nullptr;

bool initialized = false;
bool kwsPaused = false;
bool wakeVoteRing[WAKE_VOTE_WINDOW] = {false};
size_t wakeVotePos = 0;
Result lastResult{};
bool hasPendingResult = false;
uint32_t inferenceCount = 0;
char statusText[64] = "KWS not started";

float hzToMel(float hz) { return 2595.0f * log10f(1.0f + hz / 700.0f); }
float melToHz(float mel) { return 700.0f * (powf(10.0f, mel / 2595.0f) - 1.0f); }

void buildHannWindow() {
  for (size_t i = 0; i < MEL_FRAME_LEN; ++i) {
    hannWindow[i] = 0.5f - 0.5f * cosf(2.0f * kPi * static_cast<float>(i) / static_cast<float>(MEL_FRAME_LEN));
  }
}

void buildFftTwiddle() {
  for (size_t k = 0; k < MEL_FFT_LEN / 2; ++k) {
    const float angle = -2.0f * kPi * static_cast<float>(k) / static_cast<float>(MEL_FFT_LEN);
    fftTwiddleRe[k] = cosf(angle);
    fftTwiddleIm[k] = sinf(angle);
  }
}

void buildMelWeights() {
  const float lowerMel = hzToMel(kMelLowerHz);
  const float upperMel = hzToMel(kMelUpperHz);
  float hzPoints[MEL_BINS + 2];
  float fftBinHz[kFftBins];

  for (size_t i = 0; i < MEL_BINS + 2; ++i) {
    const float mel = lowerMel + (upperMel - lowerMel) * static_cast<float>(i) / static_cast<float>(MEL_BINS + 1);
    hzPoints[i] = melToHz(mel);
  }

  for (size_t bin = 0; bin < kFftBins; ++bin) {
    fftBinHz[bin] = static_cast<float>(bin) * AUDIO_SAMPLE_RATE / MEL_FFT_LEN;
  }

  for (size_t mel = 0; mel < MEL_BINS; ++mel) {
    const float leftHz = hzPoints[mel];
    const float centerHz = hzPoints[mel + 1];
    const float rightHz = hzPoints[mel + 2];
    for (size_t bin = 0; bin < kFftBins; ++bin) {
      const float hz = fftBinHz[bin];
      float weight = 0.0f;
      if (hz >= leftHz && hz < centerHz && centerHz > leftHz) {
        weight = (hz - leftHz) / (centerHz - leftHz);
      } else if (hz >= centerHz && hz < rightHz && rightHz > centerHz) {
        weight = (rightHz - hz) / (rightHz - centerHz);
      }
      melWeights[mel * kFftBins + bin] = weight;
    }
  }
}

// Ring buffer: 2 seconds of audio. micTask writes continuously,
// poll() snapshots the most recent 1 second for each inference cycle.
constexpr size_t kRingBufSamples = AUDIO_SAMPLE_RATE * 2;
constexpr int kMicSampleShift = 12;
constexpr TickType_t kPausedTaskDelay = pdMS_TO_TICKS(100);

int16_t ringBuffer[kRingBufSamples];
volatile size_t ringWritePos = 0;
volatile bool ringPrimed = false;
SemaphoreHandle_t ringMutex = nullptr;
TaskHandle_t micTaskHandle = nullptr;

bool initI2s() {
  i2s_config_t i2sConfig = {};
  i2sConfig.mode = static_cast<i2s_mode_t>(I2S_MODE_MASTER | I2S_MODE_RX);
  i2sConfig.sample_rate = AUDIO_SAMPLE_RATE;
  i2sConfig.bits_per_sample = I2S_BITS_PER_SAMPLE_32BIT;
  i2sConfig.channel_format = I2S_CHANNEL_FMT_ONLY_LEFT;
  i2sConfig.communication_format = I2S_COMM_FORMAT_STAND_I2S;
  i2sConfig.intr_alloc_flags = ESP_INTR_FLAG_LEVEL1;
  i2sConfig.dma_buf_count = 4;
  i2sConfig.dma_buf_len = 1024;
  i2sConfig.use_apll = false;
  i2sConfig.tx_desc_auto_clear = false;
  i2sConfig.fixed_mclk = 0;

  i2s_pin_config_t pinConfig = {};
  pinConfig.bck_io_num = I2S_BCK_PIN;
  pinConfig.ws_io_num = I2S_WS_PIN;
  pinConfig.data_out_num = I2S_PIN_NO_CHANGE;
  pinConfig.data_in_num = I2S_SD_PIN;

  if (i2s_driver_install(I2S_NUM_0, &i2sConfig, 0, nullptr) != ESP_OK) return false;
  if (i2s_set_pin(I2S_NUM_0, &pinConfig) != ESP_OK) return false;
  i2s_zero_dma_buffer(I2S_NUM_0);
  return true;
}

bool initInterpreter() {
  model = tflite::GetModel(g_model);
  if (model->version() != TFLITE_SCHEMA_VERSION) return false;

  static tflite::MicroMutableOpResolver<6> resolver;
  resolver.AddConv2D();
  resolver.AddDepthwiseConv2D();
  resolver.AddMaxPool2D();
  resolver.AddMean();
  resolver.AddFullyConnected();
  resolver.AddSoftmax();

  static tflite::MicroInterpreter staticInterpreter(model, resolver, tensorArena, KWS_TENSOR_ARENA_BYTES);
  interpreter = &staticInterpreter;
  if (interpreter->AllocateTensors() != kTfLiteOk) return false;

  inputTensor = interpreter->input(0);
  outputTensor = interpreter->output(0);
  if (inputTensor->bytes != MEL_N_FRAMES * MEL_BINS) return false;
  if (outputTensor->bytes != KWS_NUM_CLASSES) return false;
  return true;
}

void computeLogMel(const int16_t *samples) {
  constexpr size_t halfN = MEL_FFT_LEN / 2;
  for (size_t frame = 0; frame < MEL_N_FRAMES; ++frame) {
    const size_t offset = frame * MEL_FRAME_STEP;
    for (size_t k = 0; k < halfN; ++k) {
      fftCplx[2 * k]     = (static_cast<float>(samples[offset + 2 * k])     / 32768.0f) * hannWindow[2 * k];
      fftCplx[2 * k + 1] = (static_cast<float>(samples[offset + 2 * k + 1]) / 32768.0f) * hannWindow[2 * k + 1];
    }
    dsps_fft2r_fc32(fftCplx, halfN);
    dsps_bit_rev_fc32(fftCplx, halfN);

    const float c0re = fftCplx[0];
    const float c0im = fftCplx[1];
    fftPower[0]     = (c0re + c0im) * (c0re + c0im);
    fftPower[halfN] = (c0re - c0im) * (c0re - c0im);

    for (size_t k = 1; k < halfN; ++k) {
      const float ckRe = fftCplx[2 * k];
      const float ckIm = fftCplx[2 * k + 1];
      const float cmRe = fftCplx[2 * (halfN - k)];
      const float cmIm = fftCplx[2 * (halfN - k) + 1];

      const float aRe = 0.5f * (ckRe + cmRe);
      const float aIm = 0.5f * (ckIm - cmIm);
      const float bRe = 0.5f * (ckRe - cmRe);
      const float bIm = 0.5f * (ckIm + cmIm);

      const float twRe = fftTwiddleRe[k];
      const float twIm = fftTwiddleIm[k];
      const float wbRe = twRe * bRe - twIm * bIm;
      const float wbIm = twRe * bIm + twIm * bRe;

      const float xRe = aRe + wbIm;
      const float xIm = aIm - wbRe;
      fftPower[k] = xRe * xRe + xIm * xIm;
    }

    for (size_t mel = 0; mel < MEL_BINS; ++mel) {
      float melEnergy = 0.0f;
      const float *weights = melWeights + mel * kFftBins;
      for (size_t bin = 0; bin < kFftBins; ++bin) {
        melEnergy += fftPower[bin] * weights[bin];
      }
      featureBuffer[frame * MEL_BINS + mel] = logf(melEnergy + 1e-6f);
    }
  }
}

void quantizeInput() {
  const float scale = inputTensor->params.scale;
  const int zeroPoint = inputTensor->params.zero_point;
  for (size_t i = 0; i < MEL_N_FRAMES * MEL_BINS; ++i) {
    int value = static_cast<int>(roundf(featureBuffer[i] / scale)) + zeroPoint;
    if (value < -128) value = -128;
    if (value > 127) value = 127;
    quantizedFeature[i] = static_cast<int8_t>(value);
  }
}

bool readAudio() {
  if (!ringPrimed) {
    return false;
  }

  xSemaphoreTake(ringMutex, portMAX_DELAY);
  const size_t writePos = ringWritePos;
  size_t start;
  if (writePos >= AUDIO_DURATION_SAMPLES) {
    start = writePos - AUDIO_DURATION_SAMPLES;
  } else {
    start = kRingBufSamples - (AUDIO_DURATION_SAMPLES - writePos);
  }

  if (start + AUDIO_DURATION_SAMPLES <= kRingBufSamples) {
    memcpy(audioBuffer, ringBuffer + start, AUDIO_DURATION_SAMPLES * sizeof(int16_t));
  } else {
    const size_t firstPart = kRingBufSamples - start;
    memcpy(audioBuffer, ringBuffer + start, firstPart * sizeof(int16_t));
    memcpy(audioBuffer + firstPart, ringBuffer,
           (AUDIO_DURATION_SAMPLES - firstPart) * sizeof(int16_t));
  }
  xSemaphoreGive(ringMutex);
  return true;
}

void micTask(void *) {
  int32_t raw[512];
  while (true) {
    if (kwsPaused) {
      vTaskDelay(kPausedTaskDelay);
      continue;
    }

    size_t bytesRead = 0;
    const esp_err_t err =
        i2s_read(I2S_NUM_0, raw, sizeof(raw), &bytesRead, pdMS_TO_TICKS(1000));
    if (err != ESP_OK || bytesRead == 0) {
      vTaskDelay(pdMS_TO_TICKS(1));
      continue;
    }

    const size_t samples = bytesRead / sizeof(int32_t);
    xSemaphoreTake(ringMutex, portMAX_DELAY);
    for (size_t i = 0; i < samples; ++i) {
      int32_t sample = raw[i] >> kMicSampleShift;
      if (sample > 32767) sample = 32767;
      else if (sample < -32768) sample = -32768;
      ringBuffer[ringWritePos] = static_cast<int16_t>(sample);
      ringWritePos = (ringWritePos + 1) % kRingBufSamples;
    }
    if (!ringPrimed && ringWritePos >= AUDIO_DURATION_SAMPLES) {
      ringPrimed = true;
    }
    xSemaphoreGive(ringMutex);
  }
}

const char *findLabelByScore(float *probs, size_t *idx) {
  size_t best = 0;
  float bestScore = probs[0];
  for (size_t i = 1; i < KWS_NUM_CLASSES; ++i) {
    if (probs[i] > bestScore) {
      bestScore = probs[i];
      best = i;
    }
  }
  *idx = best;
  return kws_model_config::kClassActions[best].label;
}

}  // namespace

bool init() {
  buildHannWindow();
  buildFftTwiddle();
  buildMelWeights();
  if (dsps_fft2r_init_fc32(NULL, CONFIG_DSP_MAX_FFT_SIZE) != ESP_OK) {
    snprintf(statusText, sizeof(statusText), "FFT init failed");
    return false;
  }
  if (!initInterpreter()) {
    snprintf(statusText, sizeof(statusText), "TFLM init failed");
    return false;
  }
  if (!initI2s()) {
    snprintf(statusText, sizeof(statusText), "I2S init failed");
    return false;
  }

  ringMutex = xSemaphoreCreateMutex();
  if (!ringMutex) {
    snprintf(statusText, sizeof(statusText), "ring mutex failed");
    return false;
  }
  ringWritePos = 0;
  ringPrimed = false;
  memset(ringBuffer, 0, sizeof(ringBuffer));

  BaseType_t created = xTaskCreatePinnedToCore(
      micTask, "kws_mic", 4096, nullptr, 5, &micTaskHandle, 0);
  if (created != pdPASS) {
    snprintf(statusText, sizeof(statusText), "mic task failed");
    return false;
  }

  initialized = true;
  snprintf(statusText, sizeof(statusText), "KWS ready");
  ESP_LOGI(TAG, "KWS init done, backend=%s", kBackendMode);
  return true;
}

bool poll(Result &result) {
  if (!initialized || kwsPaused) return false;

  uint64_t t0 = esp_timer_get_time();
  if (!readAudio()) {
    uint32_t readMs = static_cast<uint32_t>((esp_timer_get_time() - t0) / 1000);
    lastResult.readMs = readMs;
    snprintf(statusText, sizeof(statusText), "ring not primed");
    return false;
  }
  uint32_t readMs = static_cast<uint32_t>((esp_timer_get_time() - t0) / 1000);

  t0 = esp_timer_get_time();
  computeLogMel(audioBuffer);
  quantizeInput();
  memcpy(inputTensor->data.int8, quantizedFeature, sizeof(quantizedFeature));
  uint32_t melMs = static_cast<uint32_t>((esp_timer_get_time() - t0) / 1000);

  t0 = esp_timer_get_time();
  if (interpreter->Invoke() != kTfLiteOk) {
    snprintf(statusText, sizeof(statusText), "Invoke failed");
    return false;
  }
  uint32_t invokeMs = static_cast<uint32_t>((esp_timer_get_time() - t0) / 1000);

  float probs[KWS_NUM_CLASSES];
  for (size_t i = 0; i < KWS_NUM_CLASSES; ++i) {
    probs[i] = (static_cast<int>(outputTensor->data.int8[i]) - outputTensor->params.zero_point) * outputTensor->params.scale;
  }

  size_t bestIdx = 0;
  const char *bestLabel = findLabelByScore(probs, &bestIdx);
  size_t top2 = bestIdx;
  size_t top3 = bestIdx;
  for (size_t i = 0; i < KWS_NUM_CLASSES; ++i) {
    if (i == bestIdx) continue;
    if (top2 == bestIdx || probs[i] > probs[top2]) top2 = i;
  }
  for (size_t i = 0; i < KWS_NUM_CLASSES; ++i) {
    if (i == bestIdx || i == top2) continue;
    if (top3 == bestIdx || top3 == top2 || probs[i] > probs[top3]) top3 = i;
  }

  const auto &action = kws_model_config::kClassActions[bestIdx];
  result.label = bestLabel;
  result.command = action.command;
  result.confidence = probs[bestIdx];
  result.wakeScore = 0.0f;
  result.hasCommand = action.command[0] != '\0' && probs[bestIdx] >= action.threshold;

  const bool wakeFrame = strcmp(bestLabel, WAKE_WORD_LABEL) == 0 && probs[bestIdx] >= action.threshold;
  wakeVoteRing[wakeVotePos] = wakeFrame;
  wakeVotePos = (wakeVotePos + 1) % WAKE_VOTE_WINDOW;
  uint8_t wakeVotes = 0;
  for (size_t i = 0; i < WAKE_VOTE_WINDOW; ++i) {
    if (wakeVoteRing[i]) ++wakeVotes;
  }
  result.isWakeWord = wakeVotes >= WAKE_VOTE_MIN;
  result.top2Label = kws_model_config::kClassActions[top2].label;
  result.top2Score = probs[top2];
  result.top3Label = kws_model_config::kClassActions[top3].label;
  result.top3Score = probs[top3];
  int16_t audioMin = 32767;
  int16_t audioMax = -32768;
  uint64_t sumSquares = 0;
  for (size_t i = 0; i < AUDIO_DURATION_SAMPLES; ++i) {
    const int16_t s = audioBuffer[i];
    if (s < audioMin) audioMin = s;
    if (s > audioMax) audioMax = s;
    sumSquares += static_cast<uint64_t>(static_cast<int32_t>(s) * static_cast<int32_t>(s));
  }
  const int32_t negMin = -static_cast<int32_t>(audioMin);
  const int32_t peak32 = audioMax > negMin ? audioMax : negMin;
  const int16_t audioPeak = peak32 > 32767 ? 32767 : static_cast<int16_t>(peak32);
  const float audioRms = sqrtf(static_cast<float>(sumSquares) / static_cast<float>(AUDIO_DURATION_SAMPLES));

  result.audioMin = audioMin;
  result.audioMax = audioMax;
  result.audioPeak = audioPeak;
  result.audioRms = audioRms;
  result.readMs = readMs;
  result.melMs = melMs;
  result.invokeMs = invokeMs;
  result.totalMs = readMs + melMs + invokeMs;
  result.inferenceCount = ++inferenceCount;

  lastResult = result;
  hasPendingResult = true;
  return hasPendingResult;
}

const char *status() { return statusText; }

const char *backendMode() { return kBackendMode; }

void setPaused(bool paused) {
  kwsPaused = paused;
  snprintf(statusText, sizeof(statusText), paused ? "KWS paused" : "KWS ready");
}

bool isPaused() { return kwsPaused; }

}  // namespace kws_engine
