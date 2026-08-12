set(PUBLIC_HEADERS
    include/util/concurrency.h
    include/util/mdarray.h
    include/util/scope_guard.h
    include/util/timers.h
    include/util/traits.h
    include/util/types.h
    include/util/unroll.h
)

add_library(util INTERFACE ${PUBLIC_HEADERS})
target_include_directories(util INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)

add_executable(util_test tests/test_types.cpp tests/concurrency_tests.cpp)
target_link_libraries(util_test PRIVATE util Catch2::Catch2WithMain)

# Doesn't work on WSL2
# target_compile_options(util_test PRIVATE -fsanitize=thread -g -O1)
# target_link_options(util_test PRIVATE -fsanitize=thread)

include(Catch)
catch_discover_tests(util_test
    TEST_PREFIX "util"
    PROPERTIES
       LABELS "util"
)