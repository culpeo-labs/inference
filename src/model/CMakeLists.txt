set(SOURCE_FILES
    src/safetensors.cpp
    src/tensor.cpp
)

set(PUBLIC_HEADERS
    include/model/safetensors.h
    include/model/tensor.h
)

add_library(model ${SOURCE_FILES} ${PUBLIC_HEADERS})
target_include_directories(model PUBLIC include)
target_link_libraries(model PRIVATE glaze::glaze)
target_link_libraries(model PUBLIC operations util)
target_compile_options(model PRIVATE -Wall -Wextra)

add_executable(tester tester.cpp)
target_link_libraries(tester model)
