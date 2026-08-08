set(SOURCES
    src/ops.cpp
)

set(PUBLIC_HEADERS
    include/operations/ops.h
)
add_library(operations ${SOURCES} ${PUBLIC_HEADER})
target_include_directories(operations PUBLIC include)
target_link_libraries(operations PUBLIC util)
target_compile_options(operations PUBLIC -mavx2 -mfma)
target_compile_options(operations PRIVATE -Wall -Wextra)


add_executable(test_ops tests/test_ops.cpp)
target_link_libraries(test_ops PRIVATE operations Catch2::Catch2WithMain)
target_include_directories(test_ops PRIVATE tests)

include(Catch)
catch_discover_tests(test_ops
  TEST_PREFIX "operations"
  PROPERTIES
    LABELS "operations"
)