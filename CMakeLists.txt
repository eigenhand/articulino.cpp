cmake_minimum_required(VERSION 3.19)
project(articulino_cpp VERSION 1.0.0 LANGUAGES C CXX)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_C_STANDARD 11)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(ARTICULINO_VULKAN "Build ggml with the Vulkan backend (experimental, the engine uses the CPU)" OFF)
option(ARTICULINO_BUILD_CLI "Build articulino-cli" ON)
option(ARTICULINO_BUILD_SHARED "Build the shared C library (libarticulino)" ON)
option(ARTICULINO_BUILD_TESTS "Build the parity test" ON)
option(ARTICULINO_NATIVE "Optimize the engine's own kernels for this CPU (-march=native), like GGML_NATIVE" ON)

if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/third_party/ggml/CMakeLists.txt")
    message(FATAL_ERROR "third_party/ggml is missing. Run: git submodule update --init --recursive")
endif()

# ggml is linked statically, so libarticulino and the tools are self-contained
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(GGML_VULKAN ${ARTICULINO_VULKAN} CACHE BOOL "" FORCE)
# ggml's own thread pool instead of OpenMP: no libgomp dependency, and as fast here (override with -DGGML_OPENMP=ON)
set(GGML_OPENMP OFF CACHE BOOL "ggml: use OpenMP")
set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_TESTS OFF CACHE BOOL "" FORCE)
add_subdirectory(third_party/ggml)

add_library(articulino_core STATIC
    src/model.cpp
    src/nn.cpp
    src/audio.cpp
    src/text.cpp
    src/synth.cpp
    src/c_api.cpp
    src/ops_cpu.cpp
)
target_include_directories(articulino_core PUBLIC include src PRIVATE third_party/pocketfft)
# only the C API (ARTICULINO_API) is exported from the shared library
set_target_properties(articulino_core PROPERTIES CXX_VISIBILITY_PRESET hidden VISIBILITY_INLINES_HIDDEN ON)
target_link_libraries(articulino_core PUBLIC ggml)
if(NOT MSVC)
    target_compile_options(articulino_core PRIVATE -Wall -Wextra -Wno-unused-parameter)
endif()
if(ARTICULINO_NATIVE AND NOT MSVC)
    target_compile_options(articulino_core PRIVATE -march=native)
elseif(NOT MSVC AND CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
    # portable builds still get the AVX2 kernels when ggml is built for AVX2 (see README)
    if(GGML_AVX2)
        target_compile_options(articulino_core PRIVATE -mavx2 -mfma -mf16c)
    endif()
endif()
find_package(Threads REQUIRED)
target_link_libraries(articulino_core PUBLIC Threads::Threads)

if(ARTICULINO_BUILD_SHARED)
    add_library(articulino SHARED src/c_api.cpp)
    target_compile_definitions(articulino PRIVATE ARTICULINO_BUILD_SHARED)
    target_link_libraries(articulino PRIVATE articulino_core)
    set_target_properties(articulino PROPERTIES PUBLIC_HEADER include/articulino.h)
    if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
        # keep the symbols of the static ggml and core libraries out of the export table
        target_link_options(articulino PRIVATE -Wl,--exclude-libs,ALL)
    endif()
endif()

if(ARTICULINO_BUILD_CLI)
    add_executable(articulino-cli apps/cli/main.cpp)
    target_link_libraries(articulino-cli PRIVATE articulino_core)
    add_executable(articulino-quantize apps/quantize/main.cpp)
    target_link_libraries(articulino-quantize PRIVATE ggml)
endif()

if(ARTICULINO_BUILD_SHARED AND ARTICULINO_BUILD_CLI)
    add_executable(stream-demo examples/stream_demo.c)
    target_link_libraries(stream-demo PRIVATE articulino)
    target_include_directories(stream-demo PRIVATE include)
endif()

if(ARTICULINO_BUILD_TESTS)
    add_executable(test-parity tests/test_parity.cpp)
    target_link_libraries(test-parity PRIVATE articulino_core)
    add_executable(test-text tests/test_text.cpp)
    target_link_libraries(test-text PRIVATE articulino_core)
endif()
