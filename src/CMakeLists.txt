set(app_sources
    ${CMAKE_SOURCE_DIR}/src/main.cpp
    ${CMAKE_SOURCE_DIR}/src/kws_engine.cpp
    ${CMAKE_SOURCE_DIR}/src/voice_service.cpp
    ${CMAKE_SOURCE_DIR}/src/tflite_builtin_code_shim.cpp
    ${CMAKE_SOURCE_DIR}/src/tflite_c_common_shim.cpp
    ${CMAKE_SOURCE_DIR}/src/tflite_kernel_util_quant_shim.cpp
)

idf_component_register(
    SRCS
        ${app_sources}
        ${CMAKE_SOURCE_DIR}/kws_voice/kws_tiny/model_tiny.cpp
    INCLUDE_DIRS
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/kws_voice/kws_tiny
        $ENV{HOME}/.platformio/lib/TensorFlowLite_ESP32/src
        $ENV{HOME}/.platformio/lib/TensorFlowLite_ESP32/src/third_party/gemmlowp
        $ENV{HOME}/.platformio/lib/TensorFlowLite_ESP32/src/third_party/gemmlowp/fixedpoint
        $ENV{HOME}/.platformio/lib/TensorFlowLite_ESP32/src/third_party/flatbuffers
        $ENV{HOME}/.platformio/lib/TensorFlowLite_ESP32/src/third_party/kissfft
        $ENV{HOME}/.platformio/lib/TensorFlowLite_ESP32/src/third_party/kissfft/tools
        $ENV{HOME}/.platformio/lib/TensorFlowLite_ESP32/src/third_party/flatbuffers/include
        $ENV{HOME}/.platformio/lib/TensorFlowLite_ESP32/src/third_party/ruy
        $ENV{HOME}/.platformio/lib/TensorFlowLite_ESP32/src/esp-nn/include
        $ENV{HOME}/.platformio/lib/TensorFlowLite_ESP32/src/esp-nn/src/common
    PRIV_REQUIRES
        tflm_micro
        tflm_signal
)

target_compile_definitions(${COMPONENT_LIB} PRIVATE TF_LITE_STATIC_MEMORY TF_LITE_DISABLE_X86_NEON)

target_compile_options(${COMPONENT_LIB} PRIVATE
    -Wno-error=attributes
    -Wno-error=shadow
    -Wno-maybe-uninitialized
    -Wno-missing-field-initializers
    -Wno-error=sign-compare
    -Wno-error=double-promotion
    -Wno-type-limits
    -Wno-error=stringop-overread
)

set_source_files_properties(${CMAKE_SOURCE_DIR}/src/kws_engine.cpp PROPERTIES COMPILE_OPTIONS "-Wno-missing-field-initializers;-Wno-cpp")

target_link_libraries(${COMPONENT_LIB} PRIVATE -lm)
