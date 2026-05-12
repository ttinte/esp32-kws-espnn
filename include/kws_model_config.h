#pragma once

#include <stddef.h>

namespace kws_model_config {

struct ClassAction {
  const char *label;
  const char *command;
  float threshold;
};

constexpr ClassAction kClassActions[] = {
    {"no", "fan_off", 0.80f},
    {"off", "light_off", 0.35f},
    {"on", "light_on", 0.35f},
    {"yes", "fan_on", 0.80f},
    {"zero", "", 0.80f},
    {"other", "", 1.10f},
};

constexpr size_t kClassCount = sizeof(kClassActions) / sizeof(kClassActions[0]);

}  // namespace kws_model_config
