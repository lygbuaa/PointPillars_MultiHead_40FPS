cmake_minimum_required(VERSION 3.10.0)
project(onnx_wrapper_demo)
set(MODULE_BIN_NAME onnx_wrapper_demo_bin)
set(MODULE_TEST_NAME onnx_wrapper_demo_test)

add_compile_options("-Wl,--no-undefined")
set(CMAKE_C_FLAGS "-fPIC -Wall -D_GNU_SOURCE -rdynamic -export-dynamic -Werror=return-type")
set(CMAKE_CXX_FLAGS "-std=c++17 -fexceptions -frtti -fPIC -Wall -D_GNU_SOURCE -rdynamic -export-dynamic -Werror=return-type")
set(CMAKE_CXX_FLAGS_RELEASE "-O2")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -fsanitize=address")

message(STATUS "PROJECT_SOURCE_DIR: ${PROJECT_SOURCE_DIR}")
message(STATUS "CMAKE_CURRENT_SOURCE_DIR: ${CMAKE_CURRENT_SOURCE_DIR}")
set(USE_ORT_VERSION 1.16.3)

set(COMMON_INC_PATH ${CMAKE_CURRENT_SOURCE_DIR}/include)
set(OPS_INC_PATH ${CMAKE_CURRENT_SOURCE_DIR}/pointpillars_ops)
set(ORT_INC_PATH ${CMAKE_CURRENT_SOURCE_DIR}/third_party/onnxruntime/${USE_ORT_VERSION}/include)
set(ORT_LIB_BASE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/third_party/onnxruntime/${USE_ORT_VERSION}/lib)
set(GTEST_INC_PATH ${CMAKE_CURRENT_SOURCE_DIR}/third_party/gtest/include)
set(GTEST_LIB_BASE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/third_party/gtest/lib)

include_directories(
    ${ORT_INC_PATH}
    ${GTEST_INC_PATH}
    ${COMMON_INC_PATH}
    ${OPS_INC_PATH}
)

file(GLOB MODULE_SRC_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/pointpillars_ops/*.cc
)
message(STATUS "MODULE_SRC_FILES: ${MODULE_SRC_FILES}")

add_executable(
  ${MODULE_BIN_NAME}
  ${MODULE_SRC_FILES}
  ${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp
)

EXECUTE_PROCESS(COMMAND uname -m COMMAND tr -d '\n' OUTPUT_VARIABLE ARCHITECTURE)
MESSAGE(STATUS "cpu architecture: ${ARCHITECTURE}")
if(${ARCHITECTURE} STREQUAL "x86_64")
  set(ORT_LIB_PATH ${ORT_LIB_BASE_PATH}/${ARCHITECTURE}/libonnxruntime.so)
  set(GTEST_LIB_PATH ${GTEST_LIB_BASE_PATH}/${ARCHITECTURE}/libgtest.so)
elseif(${ARCHITECTURE} STREQUAL "aarch64")
  set(ORT_LIB_PATH ${ORT_LIB_BASE_PATH}/${ARCHITECTURE}/libonnxruntime.so)
  set(GTEST_LIB_PATH ${GTEST_LIB_BASE_PATH}/${ARCHITECTURE}/libgtest.so)
else()
  MESSAGE(FATAL_ERROR "invalid cpu architecture!")
endif()

MESSAGE(STATUS "ORT_LIB_PATH: " ${ORT_LIB_PATH})
MESSAGE(STATUS "GTEST_LIB_PATH: " ${GTEST_LIB_PATH})

target_link_libraries(
    ${MODULE_BIN_NAME}
    ${ORT_LIB_PATH}
    yaml-cpp
    -ldl
    -lrt
    -pthread
)

option(BUILD_TEST "BUILD TESTCASES" ON)
if(BUILD_TEST)
    message(STATUS "BUILD_TEST: ${BUILD_TEST}")
    file(GLOB MODULE_TEST_FILES
      ${CMAKE_CURRENT_SOURCE_DIR}/test/*.cc
    )
    message(STATUS "MODULE_TEST_FILES: ${MODULE_TEST_FILES}")
    add_executable(
      ${MODULE_TEST_NAME}
      ${MODULE_SRC_FILES}
      ${MODULE_TEST_FILES}
    )

    target_link_libraries(
      ${MODULE_TEST_NAME}
      ${ORT_LIB_PATH}
      ${GTEST_LIB_PATH}
      yaml-cpp
      -pthread
      -ldl
      -lrt
  )
  install (TARGETS ${MODULE_TEST_NAME} DESTINATION onnx_wrapper_demo/bin/)
else()
  message(STATUS "BUILD_TEST: ${BUILD_TEST}")
endif()

install (TARGETS ${MODULE_BIN_NAME} DESTINATION onnx_wrapper_demo/bin/)
install (DIRECTORY launch DESTINATION onnx_wrapper_demo)