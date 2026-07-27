set(SOURCE_FILES
    src/config.cpp
    src/model.cpp
    src/safetensors.cpp
    src/tensor.cpp
)

set(PUBLIC_HEADERS
    include/model/model.h
    include/model/model_family.h
    include/model/safetensors.h
    include/model/tensor.h
    include/model/hf/llama_config.h
)

add_library(model ${SOURCE_FILES} ${PUBLIC_HEADERS})
target_include_directories(model PUBLIC include)
target_link_libraries(model PRIVATE glaze::glaze)
target_link_libraries(model PUBLIC operations util)
target_compile_options(model PRIVATE -Wall -Wextra)

set(TEST_FILES
    tests/model_tests.cpp
    tests/config_tests.cpp
)

add_executable(model_test ${TEST_FILES})
target_link_libraries(model_test PRIVATE model Catch2::Catch2WithMain glaze::glaze)
target_include_directories(model_test PRIVATE tests)
target_compile_definitions(model_test PRIVATE FIXTURE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/tests")

include(Catch)
catch_discover_tests(model_test
    TEST_PREFIX "model"
    PROPERTIES
       LABELS "model"
)
