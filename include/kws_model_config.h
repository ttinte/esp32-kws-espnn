#pragma once

#include <stddef.h>

namespace kws_model_config {

struct ClassAction {
  const char *label;
  const char *command;
  float threshold;
};

constexpr ClassAction kClassActions[] = {
    {"bat_den",  "light_on",  0.55f},
    {"tat_den",  "light_off", 0.55f},
    {"bat_quat", "fan_on",    0.55f},
    {"tat_quat", "fan_off",   0.55f},
    {"zero",     "",          0.65f},
    {"other",    "",          1.10f},
};

constexpr size_t kClassCount = sizeof(kClassActions) / sizeof(kClassActions[0]);

}  // namespace kws_model_config
