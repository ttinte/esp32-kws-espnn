#include <algorithm>
#include <cmath>

#include "tensorflow/lite/core/c/builtin_op_data.h"
#include "tensorflow/lite/core/c/common.h"

#include "tensorflow/lite/kernels/internal/quantization_util.h"

namespace tflite {

TfLiteStatus GetQuantizedConvolutionMultipler(TfLiteContext* context,
                                              const TfLiteTensor* input,
                                              const TfLiteTensor* filter,
                                              const TfLiteTensor* bias,
                                              TfLiteTensor* output,
                                              double* multiplier);

TfLiteStatus CalculateActivationRangeQuantized(TfLiteContext* context,
                                               TfLiteFusedActivation activation,
                                               TfLiteTensor* output,
                                               long* act_min,
                                               long* act_max);

TfLiteStatus PopulateConvolutionQuantizationParams(
    TfLiteContext* context, const TfLiteTensor* input,
    const TfLiteTensor* filter, const TfLiteTensor* bias, TfLiteTensor* output,
    const TfLiteFusedActivation& activation, long* multiplier, int* shift,
    long* output_activation_min, long* output_activation_max,
    long* per_channel_multiplier, long* per_channel_shift, int num_channels) {
  if (input == nullptr || filter == nullptr || output == nullptr ||
      multiplier == nullptr || shift == nullptr ||
      output_activation_min == nullptr || output_activation_max == nullptr ||
      per_channel_multiplier == nullptr || per_channel_shift == nullptr ||
      num_channels <= 0) {
    return kTfLiteError;
  }

  const auto* affine_quantization =
      reinterpret_cast<TfLiteAffineQuantization*>(filter->quantization.params);
  if (affine_quantization == nullptr || affine_quantization->scale == nullptr) {
    return kTfLiteError;
  }

  const bool is_per_channel = affine_quantization->scale->size > 1;
  const float* filter_scales = affine_quantization->scale->data;
  const float input_scale = input->params.scale;
  const float output_scale = output->params.scale;

  for (int i = 0; i < num_channels; ++i) {
    const float scale = is_per_channel ? filter_scales[i] : filter_scales[0];
    const double effective_output_scale =
        static_cast<double>(input_scale) * static_cast<double>(scale) /
        static_cast<double>(output_scale);
    int32_t significand = 0;
    int channel_shift = 0;
    QuantizeMultiplier(effective_output_scale, &significand, &channel_shift);
    per_channel_multiplier[i] = static_cast<long>(significand);
    per_channel_shift[i] = static_cast<long>(channel_shift);
  }

  double real_multiplier = 0.0;
  if (GetQuantizedConvolutionMultipler(context, input, filter, bias, output,
                                       &real_multiplier) != kTfLiteOk) {
    return kTfLiteError;
  }

  int exponent = 0;
  int32_t scalar_multiplier = 0;
  QuantizeMultiplier(real_multiplier, &scalar_multiplier, &exponent);
  *multiplier = static_cast<long>(scalar_multiplier);
  *shift = -exponent;

  return CalculateActivationRangeQuantized(
      context, activation, output, output_activation_min, output_activation_max);
}

TfLiteStatus PopulateConvolutionQuantizationParams(
    TfLiteContext* context, const TfLiteTensor* input,
    const TfLiteTensor* filter, const TfLiteTensor* bias, TfLiteTensor* output,
    const TfLiteFusedActivation& activation, long* multiplier, int* shift,
    long* output_activation_min, long* output_activation_max,
    long* per_channel_multiplier, long* per_channel_shift) {
  const auto* affine_quantization =
      reinterpret_cast<TfLiteAffineQuantization*>(filter->quantization.params);
  if (affine_quantization == nullptr || affine_quantization->scale == nullptr) {
    return kTfLiteError;
  }
  return PopulateConvolutionQuantizationParams(
      context, input, filter, bias, output, activation, multiplier, shift,
      output_activation_min, output_activation_max, per_channel_multiplier,
      per_channel_shift, affine_quantization->scale->size);
}


TfLiteStatus GetQuantizedConvolutionMultipler(TfLiteContext* context,
                                              const TfLiteTensor* input,
                                              const TfLiteTensor* filter,
                                              const TfLiteTensor* bias,
                                              TfLiteTensor* output,
                                              double* multiplier) {
  if (input == nullptr || filter == nullptr || output == nullptr || multiplier == nullptr) {
    return kTfLiteError;
  }

  const double input_product_scale =
      static_cast<double>(input->params.scale) * static_cast<double>(filter->params.scale);
  const double bias_scale =
      (bias != nullptr) ? static_cast<double>(bias->params.scale) : input_product_scale;
  const double output_scale = static_cast<double>(output->params.scale);

  if (output_scale == 0.0 || std::fabs(input_product_scale - bias_scale) > 1e-6) {
    return kTfLiteError;
  }

  *multiplier = input_product_scale / output_scale;
  return kTfLiteOk;
}

TfLiteStatus CalculateActivationRangeQuantized(TfLiteContext* context,
                                               TfLiteFusedActivation activation,
                                               TfLiteTensor* output,
                                               long* act_min,
                                               long* act_max) {
  if (output == nullptr || act_min == nullptr || act_max == nullptr) {
    return kTfLiteError;
  }

  const long qmin = (output->type == kTfLiteUInt8) ? 0L : -128L;
  const long qmax = (output->type == kTfLiteUInt8) ? 255L : 127L;
  const double scale = static_cast<double>(output->params.scale);
  const long zero_point = static_cast<long>(output->params.zero_point);

  switch (activation) {
    case kTfLiteActRelu:
      *act_min = std::max(qmin, static_cast<long>(std::lround(0.0 / scale) + zero_point));
      *act_max = qmax;
      break;
    case kTfLiteActRelu6:
      *act_min = std::max(qmin, static_cast<long>(std::lround(0.0 / scale) + zero_point));
      *act_max = std::min(qmax, static_cast<long>(std::lround(6.0 / scale) + zero_point));
      break;
    case kTfLiteActReluN1To1:
      *act_min = std::max(qmin, static_cast<long>(std::lround(-1.0 / scale) + zero_point));
      *act_max = std::min(qmax, static_cast<long>(std::lround(1.0 / scale) + zero_point));
      break;
    default:
      *act_min = qmin;
      *act_max = qmax;
      break;
  }

  return kTfLiteOk;
}

}  // namespace tflite
