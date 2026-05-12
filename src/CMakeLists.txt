FILE(GLOB app_sources ${CMAKE_SOURCE_DIR}/src/*.cpp)

set(TFLM_ROOT $ENV{HOME}/.platformio/lib/TensorFlowLite_ESP32/src)

FILE(GLOB tflm_sources
    ${TFLM_ROOT}/tensorflow/lite/c/*.cpp
    ${TFLM_ROOT}/tensorflow/lite/micro/*.cpp
    ${TFLM_ROOT}/tensorflow/lite/micro/arena_allocator/*.cpp
    ${TFLM_ROOT}/tensorflow/lite/micro/kernels/*.cpp
    ${TFLM_ROOT}/tensorflow/lite/micro/memory_planner/*.cpp
    ${TFLM_ROOT}/tensorflow/lite/schema/*.cpp
    ${TFLM_ROOT}/third_party/kissfft/*.c
)

list(APPEND tflm_sources
    ${TFLM_ROOT}/tensorflow/lite/core/api/error_reporter.cpp
    ${TFLM_ROOT}/tensorflow/lite/core/api/flatbuffer_conversions.cpp
    ${TFLM_ROOT}/tensorflow/lite/core/api/op_resolver.cpp
    ${TFLM_ROOT}/tensorflow/lite/kernels/internal/quantization_util.cpp
)

list(REMOVE_ITEM tflm_sources
    ${TFLM_ROOT}/tensorflow/lite/kernels/kernel_util.cpp
)

list(REMOVE_ITEM app_sources
    ${CMAKE_SOURCE_DIR}/src/tflm_kernel_util_shim.cpp
)

set(TFLM_KERNEL_UTIL_SHIM ${CMAKE_CURRENT_BINARY_DIR}/tflm_kernel_util_shim.cpp)
file(WRITE ${TFLM_KERNEL_UTIL_SHIM}
    "#include \"${TFLM_ROOT}/tensorflow/lite/kernels/kernel_util.cpp\"\n")

list(APPEND app_sources
    ${TFLM_KERNEL_UTIL_SHIM}
)

list(REMOVE_DUPLICATES app_sources)
list(REMOVE_DUPLICATES tflm_sources)

idf_component_register(
    SRCS
        ${app_sources}
        ${CMAKE_SOURCE_DIR}/kws_voice/kws_tiny/model_tiny.cpp
        ${tflm_sources}
    INCLUDE_DIRS
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/kws_voice/kws_tiny
        ${TFLM_ROOT}
)

set_source_files_properties(${TFLM_ROOT}/tensorflow/lite/micro/debug_log.cpp PROPERTIES COMPILE_DEFINITIONS TF_LITE_STRIP_ERROR_STRINGS)
set_source_files_properties(${tflm_sources} PROPERTIES COMPILE_OPTIONS "-Wno-deprecated-declarations")
set_source_files_properties(${TFLM_KERNEL_UTIL_SHIM} PROPERTIES COMPILE_OPTIONS "-Wno-deprecated-declarations")
set_source_files_properties(${CMAKE_SOURCE_DIR}/src/kws_engine.cpp PROPERTIES COMPILE_OPTIONS "-Wno-missing-field-initializers;-Wno-cpp")
