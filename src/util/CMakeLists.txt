set(PUBLIC_HEADERS
    include/util/mdarray.h
    include/util/scope_guard.h
    include/util/timers.h
    include/util/types.h
    include/util/unroll.h
)

add_library(util INTERFACE ${PUBLIC_HEADERS})
target_include_directories(util INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)

add_executable(util_test tests/test_types.cpp)
target_link_libraries(util_test PRIVATE util Catch2::Catch2WithMain)

include(Catch)
catch_discover_tests(util_test
    TEST_PREFIX "util"
    PROPERTIES
       LABELS "util"
)