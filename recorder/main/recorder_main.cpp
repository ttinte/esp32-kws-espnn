// INMP441 audio recorder — project doc lap de thu negatives ("other") qua
// dung mic thiet bi. I2S va phep >>16 phai KHOP chinh xac kws_engine.cpp de
// du lieu thu ra trung voi cai model "nhin thay" luc chay.
#include <stdio.h>

#include "driver/i2s.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "mbedtls/base64.h"

namespace {

// Khop kws_engine.cpp: pin + sample rate + dich bit.
constexpr int kI2sBckPin = 4;
constexpr int kI2sWsPin = 5;
constexpr int kI2sSdPin = 6;
constexpr int kSampleRate = 16000;
constexpr int kMicSampleShift = 16;

bool initI2s() {
  i2s_config_t cfg = {};
  cfg.mode = static_cast<i2s_mode_t>(I2S_MODE_MASTER | I2S_MODE_RX);
  cfg.sample_rate = kSampleRate;
  cfg.bits_per_sample = I2S_BITS_PER_SAMPLE_32BIT;
  cfg.channel_format = I2S_CHANNEL_FMT_ONLY_LEFT;
  cfg.communication_format = I2S_COMM_FORMAT_STAND_I2S;
  cfg.intr_alloc_flags = ESP_INTR_FLAG_LEVEL1;
  cfg.dma_buf_count = 4;
  cfg.dma_buf_len = 1024;
  cfg.use_apll = false;
  cfg.tx_desc_auto_clear = false;
  cfg.fixed_mclk = 0;

  i2s_pin_config_t pins = {};
  pins.bck_io_num = kI2sBckPin;
  pins.ws_io_num = kI2sWsPin;
  pins.data_out_num = I2S_PIN_NO_CHANGE;
  pins.data_in_num = kI2sSdPin;

  if (i2s_driver_install(I2S_NUM_0, &cfg, 0, nullptr) != ESP_OK) return false;
  if (i2s_set_pin(I2S_NUM_0, &pins) != ESP_OK) return false;
  i2s_zero_dma_buffer(I2S_NUM_0);
  return true;
}

}  // namespace

extern "C" void app_main(void) {
  if (!initI2s()) {
    printf("I2S_INIT_FAIL\n");
    return;
  }

  vTaskDelay(pdMS_TO_TICKS(300));

  int32_t raw[512];
  int16_t pcm[512];
  unsigned char b64[1400];
  uint32_t chunk = 0;

  while (true) {
    // Marker dinh ky (~2s/lan) de script khoi dong tre hoac sau reset van bat duoc.
    // PC bo qua dong nay (base64-decode that bai) nen khong lam hong audio.
    if ((chunk++ % 64) == 0) {
      printf("<<<STREAM\n");
    }
    size_t bytesRead = 0;
    if (i2s_read(I2S_NUM_0, raw, sizeof(raw), &bytesRead, portMAX_DELAY) != ESP_OK) {
      continue;
    }
    const size_t count = bytesRead / sizeof(int32_t);
    for (size_t i = 0; i < count; ++i) {
      int32_t sample = raw[i] >> kMicSampleShift;
      if (sample > 32767) sample = 32767;
      else if (sample < -32768) sample = -32768;
      pcm[i] = static_cast<int16_t>(sample);
    }

    size_t olen = 0;
    if (mbedtls_base64_encode(b64, sizeof(b64), &olen,
                              reinterpret_cast<unsigned char *>(pcm),
                              count * sizeof(int16_t)) != 0) {
      continue;
    }
    fwrite(b64, 1, olen, stdout);
    fputc('\n', stdout);
  }
}
