#pragma once

#include <stddef.h>

namespace kws_model_config {

struct ClassAction {
  const char *label;
  const char *command;
  float threshold;
};

constexpr ClassAction kClassActions[] = {
    {"bat",       "light_on",  0.55f},
    {"tat",       "light_off", 0.55f},
    {"quay",      "fan_on",    0.55f},
    {"dung",      "fan_off",   0.55f},
    {"wake_up",   "",          0.15f},
    {"other",     "",          1.10f},
};

constexpr size_t kClassCount = sizeof(kClassActions) / sizeof(kClassActions[0]);

}  // namespace kws_model_config
