#pragma once

#include <stdint.h>

namespace kws_engine {

struct Result {
  const char *label;
  const char *command;
  float confidence;
  float wakeScore;
  bool hasCommand;
  bool isWakeWord;

  const char *top2Label;
  float top2Score;
  const char *top3Label;
  float top3Score;
  int16_t audioMin;
  int16_t audioMax;
  int16_t audioPeak;
  float audioRms;

  uint32_t readMs;
  uint32_t melMs;
  uint32_t invokeMs;
  uint32_t totalMs;
  uint32_t inferenceCount;
};

bool init();
bool poll(Result &result);
const char *status();
void setPaused(bool paused);
bool isPaused();

}  // namespace kws_engine
