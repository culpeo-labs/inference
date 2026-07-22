cmake_minimum_required(VERSION 3.20)
project(infer CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(infer_ops src/ops.cpp)
target_include_directories(infer_ops PUBLIC include)
target_compile_options(infer_ops PRIVATE -Wall -Wextra)

include(FetchContent)
FetchContent_Declare(
  Catch2
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG        v3.7.1
)
FetchContent_MakeAvailable(Catch2)

# Tests run with ASan+UBSan always. Kernels index raw buffers; an OOB read
# that "works" is worse than a crash. Perf builds are a separate target
# later — never benchmark the sanitized build.
add_executable(test_ops tests/test_ops.cpp)
target_link_libraries(test_ops PRIVATE infer_ops Catch2::Catch2WithMain)
target_include_directories(test_ops PRIVATE tests)
foreach(t infer_ops test_ops)
  target_compile_options(${t} PRIVATE -fsanitize=address,undefined -g -O1)
  target_link_options(${t} PRIVATE -fsanitize=address,undefined)
endforeach()

enable_testing()
include(CTest)
include(Catch)
catch_discover_tests(test_ops)
