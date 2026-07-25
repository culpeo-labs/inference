
add_library(operations src/ops.cpp)
target_include_directories(operations PUBLIC include)
target_link_libraries(operations PUBLIC util)
target_compile_options(operations PRIVATE -Wall -Wextra)


# Tests run with ASan+UBSan always. Kernels index raw buffers; an OOB read
# that "works" is worse than a crash. Perf builds are a separate target
# later — never benchmark the sanitized build.
add_executable(test_ops tests/test_ops.cpp)
target_link_libraries(test_ops PRIVATE operations Catch2::Catch2WithMain)
target_include_directories(test_ops PRIVATE tests)
foreach(t operations test_ops)
  target_compile_options(${t} PRIVATE -fsanitize=address,undefined -g -O1)
  target_link_options(${t} PRIVATE -fsanitize=address,undefined)
endforeach()

include(Catch)
catch_discover_tests(test_ops
  TEST_PREFIX "operations"
)