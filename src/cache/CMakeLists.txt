set(PUBLIC_HEADERS
    include/cache/cache_block.h
    include/cache/kv_cache.h
)

add_library(cache INTERFACE ${SOURCE_FILES} ${PUBLIC_HEADERS})
target_include_directories(cache INTERFACE include)
target_link_libraries(cache INTERFACE util)

set(TEST_FILES
    tests/cache_block_tests.cpp
    tests/kv_cache_tests.cpp
)

add_executable(cache_test ${TEST_FILES})
target_link_libraries(cache_test PRIVATE cache Catch2::Catch2WithMain glaze::glaze)
target_include_directories(cache_test PRIVATE tests)
target_compile_definitions(cache_test PRIVATE FIXTURE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/tests")

include(Catch)
catch_discover_tests(cache_test
    TEST_PREFIX "cache"
    PROPERTIES
       LABELS "cache"
)
