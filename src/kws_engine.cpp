#include "kws_engine.h"

#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

#include <array>

#include "app_state.h"
#include "driver/gpio.h"
#include "driver/i2s.h"
#include "esp_check.h"
#include "esp_heap_caps.h"
#include "esp_idf_version.h"
#include "esp_log.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/queue.h"
#include "freertos/semphr.h"
#include "kws_model_config.h"
#include "model_tiny.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace kws_engine {
namespace {
using namespace app_state;

constexpr const char *kTag = "kws_engine";
constexpr float kPi = 3.14159265358979323846f;
constexpr size_t kFftBins = MEL_FFT_LEN / 2 + 1;
constexpr float kMelLowerHz = 20.0f;
constexpr float kMelUpperHz = 7600.0f;
constexpr TickType_t kResultQueueWait = 0;
constexpr TickType_t kPausedTaskDelay = pdMS_TO_TICKS(100);
constexpr TickType_t kIdleInferenceDelay = pdMS_TO_TICKS(20);
constexpr TickType_t kCommandInferenceDelay = pdMS_TO_TICKS(20);
constexpr int kMicSampleShift = 12;
constexpr int kMicDigitalGain = 1;
constexpr float kCommandMarginMin = 0.08f;
constexpr float kWakeVsCommandMin = 0.04f;
constexpr float kWakeTieEpsilon = 0.001f;
constexpr float kCommandVsWakeMin = 0.05f;
constexpr float kCommandVsOtherMin = 0.05f;
constexpr float kCommandTieEpsilon = 0.001f;
constexpr int16_t kHotAudioPeak = 28000;
constexpr int16_t kVeryHotAudioPeak = 31000;
constexpr float kHotAudioScale = 0.65f;
constexpr float kVeryHotAudioScale = 0.45f;
constexpr size_t kRingBufSamples = AUDIO_SAMPLE_RATE * 2;

static_assert(kws_model_config::kClassCount == KWS_NUM_CLASSES,
              "KWS_NUM_CLASSES must match kws_model_config::kClassActions");

int16_t *ringBuffer = nullptr;
volatile size_t ringWritePos = 0;
SemaphoreHandle_t ringMutex = nullptr;

int16_t *snapBuffer = nullptr;
float *hannWindow = nullptr;
float *melWeights = nullptr;
float *fftPower = nullptr;
float *fftReal = nullptr;
float *fftImag = nullptr;
float *featureBuffer = nullptr;
int8_t *quantizedFeature = nullptr;

QueueHandle_t resultQueue = nullptr;
TaskHandle_t micTaskHandle = nullptr;
TaskHandle_t inferenceTaskHandle = nullptr;

const tflite::Model *model = nullptr;
tflite::MicroInterpreter *interpreter = nullptr;
TfLiteTensor *inputTensor = nullptr;
TfLiteTensor *outputTensor = nullptr;
uint8_t *tensorArena = nullptr;

char statusText[96] = "KWS not started";
bool kwsPaused = false;
uint32_t totalInferenceCount = 0;

float hzToMel(float hz) {
  return 2595.0f * log10f(1.0f + hz / 700.0f);
}

float melToHz(float mel) {
  return 700.0f * (powf(10.0f, mel / 2595.0f) - 1.0f);
}

void fftInPlace(float *real, float *imag, size_t n) {
  size_t j = 0;
  for (size_t i = 1; i < n; ++i) {
    size_t bit = n >> 1;
    while (j & bit) {
      j ^= bit;
      bit >>= 1;
    }
    j ^= bit;
    if (i < j) {
      float tr = real[i];
      real[i] = real[j];
      real[j] = tr;
      float ti = imag[i];
      imag[i] = imag[j];
      imag[j] = ti;
    }
  }

  for (size_t len = 2; len <= n; len <<= 1) {
    const float angle = -2.0f * kPi / static_cast<float>(len);
    const float wlenR = cosf(angle);
    const float wlenI = sinf(angle);

    for (size_t i = 0; i < n; i += len) {
      float wr = 1.0f;
      float wi = 0.0f;
      for (size_t k = 0; k < len / 2; ++k) {
        const size_t u = i + k;
        const size_t v = i + k + len / 2;
        const float vr = real[v] * wr - imag[v] * wi;
        const float vi = real[v] * wi + imag[v] * wr;

        real[v] = real[u] - vr;
        imag[v] = imag[u] - vi;
        real[u] += vr;
        imag[u] += vi;

        const float nextWr = wr * wlenR - wi * wlenI;
        wi = wr * wlenI + wi * wlenR;
        wr = nextWr;
      }
    }
  }
}

void buildHannWindow() {
  for (size_t i = 0; i < MEL_FRAME_LEN; ++i) {
    hannWindow[i] = 0.5f - 0.5f * cosf(2.0f * kPi * static_cast<float>(i) /
                                       static_cast<float>(MEL_FRAME_LEN));
  }
}

void buildMelWeights() {
  const float lowerMel = hzToMel(kMelLowerHz);
  const float upperMel = hzToMel(kMelUpperHz);
  float hzPoints[MEL_BINS + 2];
  float fftBinHz[kFftBins];

  for (size_t i = 0; i < MEL_BINS + 2; ++i) {
    const float mel = lowerMel + (upperMel - lowerMel) * static_cast<float>(i) /
                                  static_cast<float>(MEL_BINS + 1);
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

void *allocKwsBuffer(size_t bytes, bool preferPsram) {
  void *ptr = nullptr;
  if (preferPsram) {
    ptr = heap_caps_malloc(bytes, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
  }
  if (!ptr) {
    ptr = heap_caps_malloc(bytes, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
  }
  return ptr;
}

bool checkBuffer(const void *ptr, const char *name, size_t bytes) {
  if (ptr != nullptr) {
    return true;
  }
  snprintf(statusText, sizeof(statusText), "%s alloc failed %u", name, static_cast<unsigned>(bytes));
  return false;
}

bool allocateBuffers() {
  ringBuffer = static_cast<int16_t *>(allocKwsBuffer(kRingBufSamples * sizeof(int16_t), true));
  if (!checkBuffer(ringBuffer, "ring", kRingBufSamples * sizeof(int16_t))) return false;
  memset(ringBuffer, 0, kRingBufSamples * sizeof(int16_t));

  snapBuffer = static_cast<int16_t *>(allocKwsBuffer(AUDIO_DURATION_SAMPLES * sizeof(int16_t), true));
  if (!checkBuffer(snapBuffer, "snap", AUDIO_DURATION_SAMPLES * sizeof(int16_t))) return false;

  hannWindow = static_cast<float *>(allocKwsBuffer(MEL_FRAME_LEN * sizeof(float), false));
  if (!checkBuffer(hannWindow, "hann", MEL_FRAME_LEN * sizeof(float))) return false;

  melWeights = static_cast<float *>(allocKwsBuffer(MEL_BINS * kFftBins * sizeof(float), false));
  if (!checkBuffer(melWeights, "melW", MEL_BINS * kFftBins * sizeof(float))) return false;

  fftPower = static_cast<float *>(allocKwsBuffer(kFftBins * sizeof(float), false));
  if (!checkBuffer(fftPower, "fftP", kFftBins * sizeof(float))) return false;

  fftReal = static_cast<float *>(allocKwsBuffer(MEL_FFT_LEN * sizeof(float), false));
  if (!checkBuffer(fftReal, "fftR", MEL_FFT_LEN * sizeof(float))) return false;

  fftImag = static_cast<float *>(allocKwsBuffer(MEL_FFT_LEN * sizeof(float), false));
  if (!checkBuffer(fftImag, "fftI", MEL_FFT_LEN * sizeof(float))) return false;

  featureBuffer = static_cast<float *>(allocKwsBuffer(MEL_N_FRAMES * MEL_BINS * sizeof(float), true));
  if (!checkBuffer(featureBuffer, "feat", MEL_N_FRAMES * MEL_BINS * sizeof(float))) return false;

  quantizedFeature = static_cast<int8_t *>(allocKwsBuffer(MEL_N_FRAMES * MEL_BINS * sizeof(int8_t), false));
  if (!checkBuffer(quantizedFeature, "qfeat", MEL_N_FRAMES * MEL_BINS * sizeof(int8_t))) return false;

  tensorArena = static_cast<uint8_t *>(allocKwsBuffer(KWS_TENSOR_ARENA_BYTES, false));
  if (!checkBuffer(tensorArena, "arena", KWS_TENSOR_ARENA_BYTES)) return false;

  return true;
}

bool initI2s() {
  const i2s_config_t i2sConfig = {
      .mode = static_cast<i2s_mode_t>(I2S_MODE_MASTER | I2S_MODE_RX),
      .sample_rate = AUDIO_SAMPLE_RATE,
      .bits_per_sample = I2S_BITS_PER_SAMPLE_32BIT,
      .channel_format = I2S_CHANNEL_FMT_ONLY_LEFT,
      .communication_format = I2S_COMM_FORMAT_STAND_I2S,
      .intr_alloc_flags = ESP_INTR_FLAG_LEVEL1,
      .dma_buf_count = 4,
      .dma_buf_len = 1024,
      .use_apll = false,
      .tx_desc_auto_clear = false,
      .fixed_mclk = 0,
#if ESP_IDF_VERSION_MAJOR >= 4
      .bits_per_chan = I2S_BITS_PER_CHAN_32BIT,
#endif
  };

  const i2s_pin_config_t pinConfig = {
      .bck_io_num = static_cast<gpio_num_t>(I2S_BCK_PIN),
      .ws_io_num = static_cast<gpio_num_t>(I2S_WS_PIN),
      .data_out_num = I2S_PIN_NO_CHANGE,
      .data_in_num = static_cast<gpio_num_t>(I2S_SD_PIN),
  };

  if (i2s_driver_install(I2S_NUM_0, &i2sConfig, 0, nullptr) != ESP_OK) {
    snprintf(statusText, sizeof(statusText), "i2s_driver_install failed");
    return false;
  }
  if (i2s_set_pin(I2S_NUM_0, &pinConfig) != ESP_OK) {
    snprintf(statusText, sizeof(statusText), "i2s_set_pin failed");
    return false;
  }
  i2s_zero_dma_buffer(I2S_NUM_0);
  return true;
}

bool initInterpreter() {
  model = tflite::GetModel(g_model);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    snprintf(statusText, sizeof(statusText), "Model schema %lu != %d",
             static_cast<unsigned long>(model->version()), TFLITE_SCHEMA_VERSION);
    return false;
  }

  static tflite::MicroMutableOpResolver<6> resolver;
  if (resolver.AddConv2D() != kTfLiteOk ||
      resolver.AddDepthwiseConv2D() != kTfLiteOk ||
      resolver.AddMaxPool2D() != kTfLiteOk ||
      resolver.AddMean() != kTfLiteOk ||
      resolver.AddFullyConnected() != kTfLiteOk ||
      resolver.AddSoftmax() != kTfLiteOk) {
    snprintf(statusText, sizeof(statusText), "TFLM resolver failed");
    return false;
  }

  static tflite::MicroInterpreter staticInterpreter(
      model, resolver, tensorArena, KWS_TENSOR_ARENA_BYTES,
      tflite::GetMicroErrorReporter());
  interpreter = &staticInterpreter;

  if (interpreter->AllocateTensors() != kTfLiteOk) {
    snprintf(statusText, sizeof(statusText), "AllocateTensors failed");
    return false;
  }

  inputTensor = interpreter->input(0);
  outputTensor = interpreter->output(0);
  if (inputTensor->bytes != MEL_N_FRAMES * MEL_BINS) {
    snprintf(statusText, sizeof(statusText), "Input bytes mismatch: %u", inputTensor->bytes);
    return false;
  }
  if (outputTensor->bytes != KWS_NUM_CLASSES) {
    snprintf(statusText, sizeof(statusText), "Output classes mismatch: model=%u fw=%u",
             outputTensor->bytes,
             static_cast<unsigned>(KWS_NUM_CLASSES));
    return false;
  }

  ESP_LOGI(kTag,
           "KWS model: %d bytes, input=%ux%u, scale=%.6f zp=%d",
           g_model_len,
           static_cast<unsigned>(MEL_N_FRAMES),
           static_cast<unsigned>(MEL_BINS),
           static_cast<double>(inputTensor->params.scale),
           inputTensor->params.zero_point);
  ESP_LOGI(kTag,
           "KWS output: classes=%u, scale=%.6f zp=%d",
           static_cast<unsigned>(outputTensor->bytes),
           static_cast<double>(outputTensor->params.scale),
           outputTensor->params.zero_point);

  return true;
}

void computeLogMel(const int16_t *samples) {
  for (size_t frame = 0; frame < MEL_N_FRAMES; ++frame) {
    const size_t offset = frame * MEL_FRAME_STEP;

    memset(fftReal, 0, MEL_FFT_LEN * sizeof(float));
    memset(fftImag, 0, MEL_FFT_LEN * sizeof(float));

    for (size_t i = 0; i < MEL_FRAME_LEN; ++i) {
      fftReal[i] = (static_cast<float>(samples[offset + i]) / 32768.0f) * hannWindow[i];
    }

    fftInPlace(fftReal, fftImag, MEL_FFT_LEN);

    for (size_t bin = 0; bin < kFftBins; ++bin) {
      fftPower[bin] = fftReal[bin] * fftReal[bin] + fftImag[bin] * fftImag[bin];
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
    if (value < -128) {
      value = -128;
    } else if (value > 127) {
      value = 127;
    }
    quantizedFeature[i] = static_cast<int8_t>(value);
  }
}

void micTask(void *) {
  int32_t raw[512];
  while (true) {
    if (kwsPaused) {
      vTaskDelay(kPausedTaskDelay);
      continue;
    }

    size_t bytesRead = 0;
    const esp_err_t err = i2s_read(
        I2S_NUM_0, raw, 256 * 2 * sizeof(int32_t), &bytesRead, pdMS_TO_TICKS(1000));
    if (err != ESP_OK || bytesRead == 0) {
      vTaskDelay(pdMS_TO_TICKS(1));
      continue;
    }

    const size_t totalSamples = bytesRead / sizeof(int32_t);
    xSemaphoreTake(ringMutex, portMAX_DELAY);
    for (size_t i = 0; i < totalSamples; ++i) {
      int32_t sample = raw[i];
      sample >>= kMicSampleShift;
      sample *= kMicDigitalGain;
      if (sample > 32767) sample = 32767;
      else if (sample < -32768) sample = -32768;
      ringBuffer[ringWritePos] = static_cast<int16_t>(sample);
      ringWritePos = (ringWritePos + 1) % kRingBufSamples;
    }
    xSemaphoreGive(ringMutex);
  }
}

void snapshotAudio() {
  xSemaphoreTake(ringMutex, portMAX_DELAY);
  size_t start;
  if (ringWritePos >= AUDIO_DURATION_SAMPLES) {
    start = ringWritePos - AUDIO_DURATION_SAMPLES;
  } else {
    start = kRingBufSamples - (AUDIO_DURATION_SAMPLES - ringWritePos);
  }

  if (start + AUDIO_DURATION_SAMPLES <= kRingBufSamples) {
    memcpy(snapBuffer, ringBuffer + start, AUDIO_DURATION_SAMPLES * sizeof(int16_t));
  } else {
    const size_t firstPart = kRingBufSamples - start;
    memcpy(snapBuffer, ringBuffer + start, firstPart * sizeof(int16_t));
    memcpy(snapBuffer + firstPart, ringBuffer, (AUDIO_DURATION_SAMPLES - firstPart) * sizeof(int16_t));
  }
  xSemaphoreGive(ringMutex);
}

bool runInference(Result &result) {
  int16_t audioMin = snapBuffer[0], audioMax = snapBuffer[0];
  uint64_t energySum = static_cast<int32_t>(snapBuffer[0]) * static_cast<int32_t>(snapBuffer[0]);
  for (size_t i = 1; i < AUDIO_DURATION_SAMPLES; ++i) {
    if (snapBuffer[i] < audioMin) audioMin = snapBuffer[i];
    if (snapBuffer[i] > audioMax) audioMax = snapBuffer[i];
    const int32_t sample = snapBuffer[i];
    energySum += static_cast<uint64_t>(sample * sample);
  }

  if (audioMin == 0 && audioMax == 0) {
    snprintf(statusText, sizeof(statusText), "Mic silence/all-zero input");
    return false;
  }

  const int16_t absMin = audioMin == INT16_MIN ? INT16_MAX : static_cast<int16_t>(abs(audioMin));
  const int16_t absMax = static_cast<int16_t>(abs(audioMax));
  const int16_t peakAbs = absMin > absMax ? absMin : absMax;
  const float audioRms = sqrtf(static_cast<float>(energySum) / static_cast<float>(AUDIO_DURATION_SAMPLES));
  float inputScale = 1.0f;
  if (peakAbs >= kVeryHotAudioPeak) {
    inputScale = kVeryHotAudioScale;
  } else if (peakAbs >= kHotAudioPeak) {
    inputScale = kHotAudioScale;
  }

  if (inputScale < 1.0f) {
    for (size_t i = 0; i < AUDIO_DURATION_SAMPLES; ++i) {
      snapBuffer[i] = static_cast<int16_t>(roundf(static_cast<float>(snapBuffer[i]) * inputScale));
    }
  }

  const uint32_t melStart = nowMs();
  computeLogMel(snapBuffer);
  quantizeInput();
  const uint32_t melMs = nowMs() - melStart;

  memcpy(inputTensor->data.int8, quantizedFeature, MEL_N_FRAMES * MEL_BINS);

  const uint32_t invokeStart = nowMs();
  if (interpreter->Invoke() != kTfLiteOk) {
    snprintf(statusText, sizeof(statusText), "Invoke failed");
    return false;
  }
  const uint32_t invokeMs = nowMs() - invokeStart;

  std::array<float, KWS_NUM_CLASSES> scores{};
  for (size_t i = 0; i < KWS_NUM_CLASSES; ++i) {
    scores[i] = (static_cast<int>(outputTensor->data.int8[i]) - outputTensor->params.zero_point) *
                outputTensor->params.scale;
  }

  int bestIndex = 0;
  for (size_t i = 1; i < KWS_NUM_CLASSES; ++i) {
    if (scores[i] > scores[bestIndex]) {
      bestIndex = static_cast<int>(i);
    }
  }

  int secondIndex = bestIndex == 0 ? 1 : 0;
  for (size_t i = 0; i < KWS_NUM_CLASSES; ++i) {
    if (static_cast<int>(i) == bestIndex) {
      continue;
    }
    if (scores[i] > scores[secondIndex]) {
      secondIndex = static_cast<int>(i);
    }
  }

  int thirdIndex = -1;
  for (size_t i = 0; i < KWS_NUM_CLASSES; ++i) {
    if (static_cast<int>(i) == bestIndex || static_cast<int>(i) == secondIndex) {
      continue;
    }
    if (thirdIndex < 0 || scores[i] > scores[thirdIndex]) {
      thirdIndex = static_cast<int>(i);
    }
  }
  if (thirdIndex < 0) {
    thirdIndex = secondIndex;
  }

  const float bestScore = scores[bestIndex];
  const float secondScore = scores[secondIndex];
  const float thirdScore = scores[thirdIndex];

  const int wakeIndex = 4;
  const int otherIndex = 5;
  const kws_model_config::ClassAction &best = kws_model_config::kClassActions[bestIndex];
  const kws_model_config::ClassAction &second = kws_model_config::kClassActions[secondIndex];
  const kws_model_config::ClassAction &third = kws_model_config::kClassActions[thirdIndex];
  const float wakeScore = scores[wakeIndex];
  const float otherScore = scores[otherIndex];

  const bool isWakeWord = std::strcmp(best.label, WAKE_WORD_LABEL) == 0 &&
                          bestScore >= best.threshold &&
                          wakeScore + kWakeTieEpsilon >= secondScore &&
                          (wakeScore - secondScore) >= kWakeVsCommandMin;

  const bool isCommandCandidate = best.command != nullptr && best.command[0] != '\0' &&
                                  bestScore >= best.threshold &&
                                  (bestScore - secondScore) >= kCommandMarginMin &&
                                  (bestScore - wakeScore) >= kCommandVsWakeMin &&
                                  (bestScore - otherScore) >= kCommandVsOtherMin;

  result.label = best.label;
  result.command = isCommandCandidate ? best.command : nullptr;
  result.confidence = bestScore;
  result.wakeScore = wakeScore;
  result.hasCommand = isCommandCandidate && !isWakeWord;
  result.isWakeWord = isWakeWord;
  result.top2Label = second.label;
  result.top2Score = secondScore;
  result.top3Label = third.label;
  result.top3Score = thirdScore;
  result.audioMin = audioMin;
  result.audioMax = audioMax;
  result.audioPeak = peakAbs;
  result.audioRms = audioRms;
  result.readMs = 0;
  result.melMs = melMs;
  result.invokeMs = invokeMs;
  result.totalMs = melMs + invokeMs;
  result.inferenceCount = ++totalInferenceCount;

  snprintf(statusText, sizeof(statusText), "ok #%lu %s %.3f",
           static_cast<unsigned long>(result.inferenceCount),
           result.label,
           static_cast<double>(result.confidence));
  return true;
}

void inferenceTask(void *) {
  Result result{};
  while (true) {
    if (kwsPaused) {
      vTaskDelay(kPausedTaskDelay);
      continue;
    }

    const uint32_t readStart = nowMs();
    snapshotAudio();
    const uint32_t readMs = nowMs() - readStart;

    if (runInference(result)) {
      result.readMs = readMs;
      xQueueOverwrite(resultQueue, &result);
    }

    vTaskDelay(wakeWordActive ? kCommandInferenceDelay : kIdleInferenceDelay);
  }
}

}  // namespace

bool init() {
  ringMutex = xSemaphoreCreateMutex();
  resultQueue = xQueueCreate(1, sizeof(Result));
  if (ringMutex == nullptr || resultQueue == nullptr) {
    snprintf(statusText, sizeof(statusText), "queue/mutex alloc failed");
    return false;
  }

  if (!allocateBuffers()) {
    return false;
  }

  buildHannWindow();
  buildMelWeights();

  if (!initI2s()) {
    return false;
  }
  if (!initInterpreter()) {
    return false;
  }

  BaseType_t micCreated = xTaskCreatePinnedToCore(micTask, "kws_mic", 4096, nullptr, 5, &micTaskHandle, 0);
  BaseType_t inferCreated = xTaskCreatePinnedToCore(inferenceTask, "kws_infer", 8192, nullptr, 5, &inferenceTaskHandle, 1);
  if (micCreated != pdPASS || inferCreated != pdPASS) {
    snprintf(statusText, sizeof(statusText), "task create failed");
    return false;
  }

  snprintf(statusText, sizeof(statusText), "KWS ready");
  return true;
}

bool poll(Result &result) {
  return xQueueReceive(resultQueue, &result, kResultQueueWait) == pdTRUE;
}

const char *status() {
  return statusText;
}

void setPaused(bool paused) {
  kwsPaused = paused;
}

bool isPaused() {
  return kwsPaused;
}

}  // namespace kws_engine
